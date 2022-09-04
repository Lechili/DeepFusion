import warnings
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def check_gospa_parameters(c, p, alpha):
    """ Check parameter bounds.
    If the parameter values are outside the allowable range specified in the
    definition of GOSPA, a ValueError is raised.
    """
    if alpha <= 0 or alpha > 2:
        raise ValueError("The value of alpha is outside the range (0, 2]")
    if c <= 0:
        raise ValueError("The cutoff distance c is outside the range (0, inf)")
    if p < 1:
        raise ValueError("The order p is outside the range [1, inf)")

class MotLoss(nn.Module):
    def __init__(self, params):
        super().__init__()
        if params.loss.type == 'gospa':
            check_gospa_parameters(params.loss.cutoff_distance, params.loss.order, params.loss.alpha)
            self.order = params.loss.order
            self.cutoff_distance = params.loss.cutoff_distance
            self.alpha = params.loss.alpha
            self.miss_cost = self.cutoff_distance ** self.order
        self.params = params
        self.device = torch.device(params.training.device)
        self.to(self.device)

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i)
                              for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i)
                              for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def compute_hungarian_matching(self, predicted_states, predicted_logits, targets, distance='detr', scaling=1):
        """ Performs the matching

        Params:
            outputs: dictionary with 'state' and 'logits'
                state: Tensor of dim [batch_size, num_queries, d_label]
                logits: Tensor of dim [batch_size, num_queries, number_of_classes]

            targets: This is a list of targets (len(targets) = batch_size), where each target is a
                    tensor of dim [num_objects, d_label] (where num_objects is the number of ground-truth
                    objects in the target)

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        bs, num_queries = predicted_states.shape[:2]
        predicted_probabilities = predicted_logits.sigmoid().flatten(0,1)

        out = predicted_states.flatten(0, 1)
        tgt = torch.cat(targets)

        cost = torch.cdist(out, tgt, p=2)
        cost -= predicted_probabilities.log()
        cost = cost.view(bs, num_queries, -1).cpu()
        sizes = [len(v) for v in targets]

        # Perform hungarian matching using scipy linear_sum_assignment
        with torch.no_grad():
            indices = [linear_sum_assignment(
                c[i]) for i, c in enumerate(cost.split(sizes, -1))]
            permutation_idx = [(torch.as_tensor(i, dtype=torch.int64).to(torch.device(self.device)), torch.as_tensor(
                j, dtype=torch.int64).to(self.device)) for i, j in indices]
        return permutation_idx, cost.to(self.device)

    def compute_orig_gospa_matching(self, outputs, targets, existence_threshold):
        """ Performs the matching. Note that this can NOT be used as a loss function
        Params:
            outputs: dictionary with 'state' and 'logits'
                state: Tensor of dim [batch_size, num_queries, d_label]
                logits: Tensor of dim [batch_size, num_queries, number_of_classes]
            targets: This is a list of targets (len(targets) = batch_size), where each target is a
                    tensor of dim [num_objects, d_label] (where num_objects is the number of ground-truth
                    objects in the target)
            existence_threshold: Float in range (0,1) that decides which object are considered alive and which are not.
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        assert 'state' in outputs, "'state' should be in dict"
        assert 'logits' in outputs, "'logits' should be in dict"
        assert self.order == 1, 'This code does not work for loss.order != 1'
        assert self.alpha == 2, 'The permutation -> assignment relation used to decompose GOSPA might require that loss.alpha == 2'

        output_state = outputs['state'].detach()
        output_existence_probabilities = outputs['logits'].sigmoid().detach()

        bs, num_queries = output_state.shape[:2]
        dim_predictions = output_state.shape[2]
        dim_targets = targets[0].shape[1]
        assert dim_predictions == dim_targets

        loss = torch.zeros(size=(1,))
        localization_cost = 0
        missed_target_cost = 0
        false_target_cost = 0
        indices = []

        for i in range(bs):
            alive_idx = output_existence_probabilities[i, :].squeeze(-1) > existence_threshold
            alive_output = output_state[i, alive_idx, :]
            current_targets = targets[i]
            permutation_length = 0

            if len(current_targets) == 0:
                indices.append(([], []))
                loss += torch.Tensor([self.miss_cost/self.alpha * len(alive_output)])
                false_target_cost = self.miss_cost/self.alpha * len(alive_output)
            elif len(alive_output) == 0:
                indices.append(([], []))
                loss += torch.Tensor([self.miss_cost/self.alpha * len(current_targets)])
                missed_target_cost = self.miss_cost / self.alpha * len(current_targets)
            else:
                dist = torch.cdist(alive_output, current_targets, p=2)
                dist = dist.clamp_max(self.cutoff_distance)
                c = torch.pow(input=dist, exponent=self.order)
                c = c.cpu()
                output_idx, target_idx = linear_sum_assignment(c)
                indices.append((output_idx, target_idx))

                for t, o in zip(output_idx, target_idx):
                    loss += c[t,o]
                    if c[t, o] < self.cutoff_distance:
                        localization_cost += c[t, o].item()
                        permutation_length += 1

                cardinality_error = abs(len(alive_output) - len(current_targets))
                loss += self.miss_cost/self.alpha * cardinality_error

                missed_target_cost += (len(current_targets) - permutation_length) * (self.miss_cost/self.alpha)
                false_target_cost += (len(alive_output) - permutation_length) * (self.miss_cost/self.alpha)

        decomposition = {'localization': localization_cost, 'missed': missed_target_cost, 'false': false_target_cost,
                         'n_matched_objs': permutation_length}
        return loss, indices, decomposition

    def state_loss(self, predicted_states, targets, indices, uncertainties=None):
        idx = self._get_src_permutation_idx(indices)
        matched_predicted_states = predicted_states[idx]
        target = torch.cat([t[i] for t, (_, i) in zip(targets, indices)], dim=0)
        if uncertainties is not None:
            matched_uncertainties = uncertainties[idx]
            prediction_distribution = torch.distributions.normal.Normal(matched_predicted_states, matched_uncertainties)
            loss = -prediction_distribution.log_prob(target).mean()
        else:
            loss = F.l1_loss(matched_predicted_states, target, reduction='none').sum(-1).mean()

        return loss

    def logits_loss(self, predicted_logits, targets, indices):
        idx = self._get_src_permutation_idx(indices)

        target_classes = torch.zeros_like(predicted_logits, device=predicted_logits.device)
        target_classes[idx] = 1.0  # this is representation of an object
        loss = F.binary_cross_entropy_with_logits(predicted_logits.squeeze(-1).permute(1,0), target_classes.squeeze(-1).permute(1,0))

        return loss

    def get_loss(self, prediction, targets, loss_type):
        # Create state vectors for the predictions, based on prediction target specified by user
        if self.params.data_generation.prediction_target == 'position':
            predicted_states = prediction.positions
        elif self.params.data_generation.prediction_target == 'position_and_velocity':
            predicted_states = torch.cat((prediction.positions, prediction.velocities), dim=2)
        else:
            raise NotImplementedError(f'Hungarian matching not implemented for prediction target '
                                      f'{self.params.data_generation.prediction_target}')

        indices, _ = self.compute_hungarian_matching(predicted_states, prediction.logits, targets)
        log_loss = self.logits_loss(prediction.logits, targets, indices)
        if hasattr(prediction, 'uncertainties'):
            state_loss = self.state_loss(predicted_states, targets, indices, uncertainties=prediction.uncertainties)
        else:
            state_loss = self.state_loss(predicted_states, targets, indices)
        loss = {f'{loss_type}_state': state_loss, f'{loss_type}_logits': log_loss}

        return loss, indices

    def forward(self, targets, prediction, intermediate_predictions=None, encoder_prediction=None, loss_type='detr'):
        if loss_type not in [ 'detr']:
            raise NotImplementedError(f"The loss type '{loss_type}' was not implemented.'")

        losses = {}
        loss, indices = self.get_loss(prediction, targets, loss_type)
        losses.update(loss)

        if intermediate_predictions is not None:
            for i, intermediate_prediction in enumerate(intermediate_predictions):
                aux_loss, _ = self.get_loss(intermediate_prediction, targets, loss_type)
                aux_loss = {f'{k}_{i}': v for k, v in aux_loss.items()}
                losses.update(aux_loss)

        if encoder_prediction is not None:
            enc_loss, _ = self.get_loss(encoder_prediction, targets, loss_type)
            enc_loss = {f'{k}_enc': v for k, v in enc_loss.items()}
            losses.update(enc_loss)

        return losses, indices
