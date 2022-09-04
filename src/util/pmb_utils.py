import numpy as np
import torch
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from scipy.optimize import linear_sum_assignment
from util.misc import AnnotatedValue, AnnotatedValueSum


def compute_nll_for_pmb(predictions, targets, target_infos=None):
    test_score = (2**53-1)/2**53
    if target_infos is None:
        dummy_ids = [0 for _ in range(len(targets))]
        dummy_trajectories = [[[-1, -1, -1, -1, -1]]]
        target_infos = [dummy_ids, dummy_trajectories]
    target_ids, all_trajectories = target_infos

    n_predictions = len(predictions['means'])
    n_targets = len(targets)

    if len(predictions['covs'].shape) == 3:
#        print('SHAPE 3 ---------------------------------------------------------')
        distribution_type = MultivariateNormal
        scale_params = predictions['covs']
    else:
        distribution_type = Normal
        scale_params = predictions['covs'].sqrt()

    cost_matrix = np.ones((n_predictions + n_targets, n_targets)) * np.inf
    for i_prediction in range(n_predictions):
        p_existence = predictions['p_exs'][i_prediction].item()
        if p_existence >= 1:
            p_existence = test_score
        elif p_existence <= 0:
            p_existence = 1- test_score
        else:
            p_existence = p_existence


        dist = distribution_type(predictions['means'][i_prediction],scale_params[i_prediction])

        for i_target in range(n_targets):
            cost_matrix[i_prediction, i_target] = -(np.log(p_existence) + dist.log_prob(targets[i_target]).sum() - np.log(1-p_existence))
    # Fill in diagonal of sub-matrix corresponding to PPP matches
    for i_target in range(n_targets):
        temp = predictions['ppp_log_prob_func'](targets[i_target])
        if temp == 0:
            cost_matrix[n_predictions + i_target, i_target] = - (1-test_score)
        else:
            cost_matrix[n_predictions + i_target, i_target] = -temp


    # Find optimal match using Hungarian algorithm
    optimal_match = linear_sum_assignment(cost_matrix)

    # Compute likelihood and decompositions
    annotated_cost = AnnotatedValueSum()

    annotated_cost.add(AnnotatedValue(predictions['ppp_lambda'], {'type': 'miss'})) # Integral in the missed term

    for i_prediction, i_target in zip(optimal_match[0], optimal_match[1]):
        birth_time_annotation = all_trajectories[target_ids[i_target]][0][4]
        # For targets matched with predictions, add cost for localization and existence probability
        if i_prediction < n_predictions:
            p_existence = predictions['p_exs'][i_prediction].item()
            if p_existence >= 1:
                p_existence = test_score
            elif p_existence <= 0:
                p_existence = 1- test_score
            else:
                p_existence = p_existence
            temp = -np.log(p_existence) + np.log(1-p_existence)
            annotated_cost.add(AnnotatedValue(cost_matrix[i_prediction, i_target] - temp,
                                              {'type': 'loc',
                                               'target_state': targets[i_target],
                                               'target_birth_time': birth_time_annotation}))
            annotated_cost.add(AnnotatedValue(-np.log(p_existence),
                                              {'type': 'p_true',
                                               'target_state': targets[i_target],
                                               'target_birth_time': birth_time_annotation}))

        # For targets matched with PPP, just add cost for explaining missed targets
        else:
            annotated_cost.add(AnnotatedValue(cost_matrix[i_prediction, i_target],
                                              {'type': 'miss',
                                               'target_state': targets[i_target],
                                               'target_birth_time': birth_time_annotation}))
    # Afterwards, add -log(1-p) for all predictions false predictions.
    for i_prediction in range(n_predictions):
        if i_prediction not in optimal_match[0]:
            p_existence = predictions['p_exs'][i_prediction].item()
            if p_existence >= 1:
                p_existence = test_score
            elif p_existence <= 0:
                p_existence = 1- test_score
            else:
                p_existence = p_existence
            annotated_cost.add(AnnotatedValue(-np.log(1-p_existence),
                                              {'type': 'p_false', 'pred_state': predictions['means'][i_prediction]}))

#    print('The values are',annotated_cost.values)
#    print('The corresponding annotations are', annotated_cost.annotations)

    negative_log_likelihood = annotated_cost.get_total_value()

#    print('By summing all the values in annotated cost, the nll is',negative_log_likelihood)

#    print('The loc value before sum is',annotated_cost.get_filtered_values(lambda x: x.get('type')=='loc'))
    loc_cost = sum(annotated_cost.get_filtered_values(lambda x: x.get('type')=='loc'))

#    print('The p_true value before sum is',annotated_cost.get_filtered_values(lambda x: x.get('type')=='p_true'))
    p_true_cost = sum(annotated_cost.get_filtered_values(lambda x: x.get('type')=='p_true'))

#    print('The p_false value before sum is',annotated_cost.get_filtered_values(lambda x: x.get('type')=='p_false'))
    p_false_cost = sum(annotated_cost.get_filtered_values(lambda x: x.get('type')=='p_false'))

#    print('The miss value before sum is',annotated_cost.get_filtered_values(lambda x: x.get('type')=='miss'))
    p_miss_cost = sum(annotated_cost.get_filtered_values(lambda x: x.get('type')=='miss'))

    return negative_log_likelihood, (loc_cost,p_true_cost, p_false_cost, p_miss_cost), optimal_match, annotated_cost

class UnnormalizedGaussianMixture:
    def __init__(self, weights, means, covs):
        self.weights = weights
        self.components = MultivariateNormal(means, covs)

    def log_prob(self, x):
        temp = 0
        log_probs_for_each_component = self.components.log_prob(x)
        log_probs_for_each_weight = self.weights.log()
        for i in range(len(self.weights)):
            temp = temp + torch.exp(log_probs_for_each_component[i] + log_probs_for_each_weight[i])

        return temp.log()

    def get_lambda(self):
        return self.weights.sum().item()
#class UnnormalizedGaussianMixture:
#    def __init__(self, weights, means, covs):
#        self.weights = weights
#        self.components = MultivariateNormal(means, covs)

#    def log_prob(self, x):
#        log_probs_for_each_component = self.components.log_prob(x)
#        return torch.logsumexp(log_probs_for_each_component + self.weights.log(), 0).item()

#    def get_lambda(self):
#        return self.weights.sum().item()
