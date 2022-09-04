import torch
from modules.loss import MotLoss

def evaluate_gospa(data_generator, model, eval_params, params, EVAL):
    with torch.no_grad():
        model.eval()
        mot_loss = MotLoss(eval_params)
        gospa_total = 0
        gospa_loc = 0
        gospa_norm_loc = 0
        gospa_miss = 0
        gospa_false = 0

        #batch, labels = data_generator.get_batch(eval_data,eval_labels)
        #prediction, intermediate_predictions, encoder_prediction, aux_classifications, _ = model.forward(batch)
        #loss_dict, _ = mot_loss.forward(labels, prediction, intermediate_predictions, encoder_prediction, loss_type='detr')
        #model.train()
        eval_data = []
        eval_labels = []

        for i_ev in range(eval_params.n_samples):
            if EVAL['ES'][i_ev][0][0] is not None and EVAL['ES'][i_ev][0][0].any() :
                if EVAL['ES'][i_ev][0][0].ndim == 1:
                    eval_data.append(EVAL['ES'][i_ev][0][0].reshape(1,params.loss.vector_length))
                else:
                    eval_data.append(EVAL['ES'][i_ev][0][0])

                if EVAL['GT'][i_ev][0][0] is not None and EVAL['GT'][i_ev][0][0].any() :

                    if EVAL['GT'][i_ev][0][0].ndim == 1:
                        eval_labels.append(EVAL['GT'][i_ev][0][0].reshape(1,4))

                    else:
                        eval_labels.append(EVAL['GT'][i_ev][0][0])

                else:
                    eval_labels.append([])
        print(eval_data)
        #Get batch from data generator and feed it to trained model
        batch, labels = data_generator.get_batch(eval_data,eval_labels)
        prediction, _, _, _, _ = model.forward(batch)

        # Compute GOSPA score
        prediction_in_format_for_loss = {'state': torch.cat((prediction.positions, prediction.velocities), dim=2),
                                         'logits': prediction.logits,
                                         'state_covariances': prediction.uncertainties ** 2}
        loss, _, decomposition = mot_loss.compute_orig_gospa_matching(prediction_in_format_for_loss, labels,
                                                                      eval_params.loss.existence_prob_cutoff)
        gospa_total += loss.item()
        gospa_loc += decomposition['localization']
        gospa_norm_loc += decomposition['localization'] / decomposition['n_matched_objs'] if \
            decomposition['n_matched_objs'] != 0 else 0.0
        gospa_miss += decomposition['missed']
        gospa_false += decomposition['false']

        model.train()
        gospa_total /= eval_params.n_samples
        gospa_loc /= eval_params.n_samples
        gospa_norm_loc /= eval_params.n_samples
        gospa_miss /= eval_params.n_samples
        gospa_false /= eval_params.n_samples
    return gospa_total, gospa_loc, gospa_norm_loc, gospa_miss, gospa_false
