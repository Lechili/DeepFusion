
from util.misc import super_load
from util.load_config_files import load_yaml_into_dotdict
import argparse
import warnings


# Parse arguments and load the model, before doing anything else (important, reduces possibility of weird bugs)
parser = argparse.ArgumentParser()
parser.add_argument('-rp', '--result_filepath', help='filepath to result folder for trained model', required=True)
parser.add_argument('-tp', '--task_params', help='filepath to configuration yaml file defining the task', required=False)
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



import os
import pickle

import numpy as np
from data_generation.my_data_generation import DataGenerator
import mat73
from scipy.io import savemat

EVAL = mat73.loadmat(params.dataset.test_data)
# Read evaluation hyperparameters and overwrite `params` with them
eval_params = load_yaml_into_dotdict('configs/eval/default.yaml')
params.recursive_update(eval_params)

data_generator = DataGenerator(params)


eval_dict = {}

for i_eval_step in range(1000):

    eval_data = []
    eval_labels = []
    for i in range(1):

        if EVAL['ES'][i_eval_step][0][0] is not None and EVAL['ES'][i_eval_step][0][0].any() :
            if EVAL['ES'][i_eval_step][0][0].ndim == 1:
                eval_data.append(EVAL['ES'][i_eval_step][0][0].reshape(1,params.loss.vector_length))
            else:
                eval_data.append(EVAL['ES'][i_eval_step][0][0])


            if EVAL['GT'][i_eval_step][0][0] is not None and EVAL['GT'][i_eval_step][0][0].any() :
                if EVAL['GT'][i_eval_step][0][0].ndim == 1:
                    eval_labels.append(EVAL['GT'][i_eval_step][0][0].reshape(1,4))
                else:
                    eval_labels.append(EVAL['GT'][i_eval_step][0][0])
            else:
                eval_labels.append([])

        packed_eval_batch, packed_eval_labels = data_generator.get_batch(eval_data,eval_labels)
        outputs, _, _, _,attn_maps = model.forward(packed_eval_batch)

        attn_weights = attn_maps['intermediate_attention'][-1].cpu().detach().numpy()

        output_positions = outputs.positions.cpu().detach().numpy()

        output_velocities = outputs.velocities.cpu().detach().numpy()

        output_logits = outputs.logits.sigmoid().cpu().detach().numpy()

        output_uncertainties = outputs.uncertainties.cpu().detach().numpy()

        out_together = np.concatenate((output_positions,output_velocities,output_uncertainties,output_logits),axis = 2)
        alive_output = out_together[0, :, :]

        eval_dict['eval'+str(i_eval_step)] = np.concatenate((alive_output,attn_weights[0,:,:]),axis = 1)
print('Finish evaluation')
if params.arch.use_nerf:
    exp_name = params.dataset.scenario + '_' + params.dataset.training_steps+ '_' + params.dataset.task + '_' + 'lrp' + '_' + str(int(params.training.reduce_lr_patience/1000)) +'k'+'_'+'rn' + '_' + str(params.arch.multires) + '_'+params.dataset.predicts
else:
    exp_name = params.dataset.scenario + '_' + params.dataset.training_steps + '_'+params.dataset.task + '_' + 'lrp' + '_' + str(int(params.training.reduce_lr_patience/1000))  +'k'+ '_'+params.dataset.predicts
savemat('test_mt3v2_' + exp_name + '.mat',eval_dict)
