"""
Script that helps tune a trained MT3v2 model's PPP intensity. The result path is specified via -rp argument,
and its predictions are computed on the fly by the script. Make sure to tune the PPP intensity using a different seed
than the one used for evaluation later.
"""


from util.misc import super_load
from util.load_config_files import load_yaml_into_dotdict
import argparse
import warnings


# Parse arguments and load the model, before doing anything else (important, reduces possibility of weird bugs)
parser = argparse.ArgumentParser()
parser.add_argument('-rp', '--result_filepath', help='filepath to result folder for trained model', required=True)
parser.add_argument('-tp', '--task_params', help='filepath to configuration yaml file defining the task', required=True)
parser.add_argument('--show_debugging_plots', help='Flag to decide if samples will be plotted or not', action='store_true', required=False)
parser.add_argument('--plot_decomposition_for_all_samples', action='store_true', required=False)
args = parser.parse_args()
print(f'Evaluating results from folder: {args.result_filepath}...')
model, params = super_load(args.result_filepath, verbose=True)

# Test that the model was trained in the task chosen for evaluation
if args.task_params is not None:
    task_params = load_yaml_into_dotdict(args.task_params)
    for k, v in task_params.data_generation.items():
        if k not in params.data_generation:
            warnings.warn(f"Key '{k}' not found in trained model's hyperparameters")
        elif params.data_generation[k] != v:
            warnings.warn(f"Different values for key '{k}'. Task: {v}\tTrained: {params.data_generation[k]}")
    # Use task params, not the ones from the trained model
    params.recursive_update(task_params)  # note: parameters specified only on trained model will remain untouched
else:
    warnings.warn('Evaluation task was not specified; inferring it from the task specified in the results folder.')


import time
import torch
from torch import Tensor
import numpy as np
#import matplotlib.pyplot as plt

from modules.loss import MotLoss
from data_generation.my_data_generation import DataGenerator
import mat73
from util.pmb_utils import compute_nll_for_pmb,UnnormalizedGaussianMixture
EVAL = mat73.loadmat(params.dataset.test_data)

# Read evaluation hyperparameters and overwrite `params` with them
eval_params = load_yaml_into_dotdict('configs/eval/default.yaml')
params.recursive_update(eval_params)

# Check this script is running on correct settings
#    raise NotImplementedError('This script only works when models are predicting position and velocity estimates.')
#if params.training.batch_size != 1:
#    raise NotImplementedError('This script only works for batch size == 1')

mot_loss = MotLoss(params)
data_generator = DataGenerator(params)
#lambda_values = np.linspace(0.001, 0.02, 10)  # Change these while tuning

#prepare the SOTA data from MATLAB
#results = []

with torch.no_grad():
#    for lambda_value in lambda_values:

    mt3_nll_results = {'total': [], 'loc': [], 'p_false': [], 'miss': []}
    sota_nll_results = {'total': [], 'loc': [], 'p_false': [], 'miss': []}

    lam = 0.0001 
    #t = time.time()
    #print(f"λ: {lambda_value}")

    for i in range(1000):
        eval_data = []
        eval_labels = []

        if EVAL['ES'][i][0][0] is not None and EVAL['ES'][i][0][0].any() :
            if EVAL['ES'][i][0][0].ndim == 1:
                eval_data.append(EVAL['ES'][i][0][0].reshape(1,params.loss.vector_length))
            else:
                eval_data.append(EVAL['ES'][i][0][0])


            if EVAL['GT'][i][0][0] is not None and EVAL['GT'][i][0][0].any() :
                if EVAL['GT'][i][0][0].ndim == 1:
                    eval_labels.append(EVAL['GT'][i][0][0].reshape(1,4))
                else:
                    eval_labels.append(EVAL['GT'][i][0][0])
            else:
                eval_labels.append([])

        # Get batch from data generator and feed it to trained model
        batch, labels= data_generator.get_batch(eval_data,eval_labels)
        prediction, _, _, _, _ = model.forward(batch)

#        print('The labels are',labels)
#        print('The labels[0] are', labels[0])

        # compute NLL for MT3v2
        predictions_mt3 = {'state': torch.cat((prediction.positions, prediction.velocities), dim=2),
                           'logits': prediction.logits,
                           'state_covariances': prediction.uncertainties**2}

        predicted_pmb_mt3 = {'means': predictions_mt3['state'][0],
                             'covs': prediction.uncertainties[0]**2,
                             'p_exs': torch.sigmoid(prediction.logits[0]),
                             'ppp_lambda': lam,
                             'ppp_log_prob_func': lambda x: np.log(lam/400)}

        #print('The predicted_pmb_mt3 is,',predicted_pmb_mt3)
        #print('The predicted_mt3 is,',predictions_mt3)

        nll_sample_mt3, nll_decomposition_mt3, _, _ = compute_nll_for_pmb(predicted_pmb_mt3, labels[0])

        # compute NLL for sota


        s1 = EVAL['PPP'][i][0][0]['weight']

        s2 = EVAL['PPP'][i][0][0]['mean']

        s3 = EVAL['PPP'][i][0][0]['covariance']
        if np.size(s1) == 1:
            s1 = [s1]
            s2 = [s2]
            s3 = [np.array(s3)]

        s4 = EVAL['SOTA'][i][0][0]['weight']

        s5 = EVAL['SOTA'][i][0][0]['mean']

        s6 = EVAL['SOTA'][i][0][0]['covariance']



        ppp_weight = np.zeros((len(s1),1))
        ppp_mean = np.zeros((len(s1),4))
        ppp_cov = np.zeros((len(s1),4,4))

        sota_weight = np.zeros((len(s4),1))
        sota_mean = np.zeros((len(s4),4))
        sota_cov = np.zeros((len(s4),4,4))

        for j in range(len(s1)):
            ppp_weight[j,:] = s1[j]
            ppp_mean[j,:] = s2[j]
            ppp_cov[j,:,:] = s3[j]


        for j in range(len(s4)):
            sota_weight[j,:] = s4[j]
            sota_mean[j,:] = s5[j]
            sota_cov[j,:,:] = s6[j]

        ppp_weight = Tensor(list(ppp_weight)).to(torch.device('cuda'))
        ppp_mean = Tensor(list(ppp_mean)).to(torch.device('cuda'))
        ppp_cov = Tensor(list(ppp_cov)).to(torch.device('cuda'))

        sota_weight = Tensor(list(sota_weight)).to(torch.device('cuda'))
        sota_mean = Tensor(list(sota_mean)).to(torch.device('cuda'))
        sota_cov = Tensor(list(sota_cov)).to(torch.device('cuda'))

        PPP = UnnormalizedGaussianMixture(ppp_weight,ppp_mean,ppp_cov)

        predicted_pmb_sota = {'means': sota_mean,
                             'covs': sota_cov,
                             'p_exs': sota_weight,
                             'ppp_lambda': PPP.get_lambda(),
                             'ppp_log_prob_func': lambda x: PPP.log_prob(x)}

        nll_sample_sota, nll_decomposition_sota, _, _ = compute_nll_for_pmb(predicted_pmb_sota, labels[0])

        # Save results from this sample
        mt3_nll_results['total'].append(nll_sample_mt3)
        mt3_nll_results['loc'].append(nll_decomposition_mt3[0] + nll_decomposition_mt3[1])
        mt3_nll_results['p_false'].append(nll_decomposition_mt3[2])
        mt3_nll_results['miss'].append(nll_decomposition_mt3[3])

        sota_nll_results['total'].append(nll_sample_sota)
        sota_nll_results['loc'].append(nll_decomposition_sota[0]+nll_decomposition_sota[1])
        sota_nll_results['p_false'].append(nll_decomposition_sota[2])
        sota_nll_results['miss'].append(nll_decomposition_sota[3])

        if (i % (1 + (1000 // 10))) == 0:
            print(f"\tProcessed {i + 1}/{1000}...")

    mt3_average_nll_total = np.mean(mt3_nll_results['total'])
    mt3_average_nll_loc = np.mean(mt3_nll_results['loc'])
    mt3_average_nll_p_false = np.mean(mt3_nll_results['p_false'])
    mt3_average_nll_miss = np.mean(mt3_nll_results['miss'])

    sota_average_nll_total = np.mean(sota_nll_results['total'])
    sota_average_nll_loc = np.mean(sota_nll_results['loc'])
    #sota_average_nll_p_true = np.mean(sota_nll_results['p_true'])
    sota_average_nll_p_false = np.mean(sota_nll_results['p_false'])
    sota_average_nll_miss = np.mean(sota_nll_results['miss'])

    print(' ------------------ MT3v2 NLL scores ----------------------')
    print('Average NLL total:',mt3_average_nll_total)
    print('Average NLL loc:', mt3_average_nll_loc)
    print('Average NLL p_miss:',mt3_average_nll_miss)
    print('Average NLL p_false:',mt3_average_nll_p_false)


    print(' ------------------ SOTA NLL scores ----------------------')
    print('Average NLL total:',sota_average_nll_total)
    print('Average NLL loc:', sota_average_nll_loc)
    print('Average NLL p_miss:',sota_average_nll_miss)
    print('Average NLL p_false:',sota_average_nll_p_false)
