#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import datetime
import re
import shutil
from collections import deque
import argparse

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from data_generation.my_data_generation import DataGenerator
from util.misc import save_checkpoint, update_logs#
from util.load_config_files import load_yaml_into_dotdict

from util.logger import Logger
from modules.loss import MotLoss

from modules import evaluator
from modules.models.mt3v2.mt3v2 import MT3V2
import mat73
from scipy.io import savemat

if __name__ == '__main__':

    # Load CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-tp', '--task_params', help='filepath to configuration yaml file defining the task', required=True)
    parser.add_argument('-mp', '--model_params', help='filepath to configuration yaml file defining the model', required=True)
    parser.add_argument('--continue_training_from', help='filepath to folder of an experiment to continue training from')
    parser.add_argument('--exp_name', help='Name to give to the results folder')
    args = parser.parse_args()
    print(f'Task configuration file: {args.task_params}')
    print(f'Model configuration file: {args.model_params}')

    # Load hyperparameters
    params = load_yaml_into_dotdict(args.task_params)
    params.update(load_yaml_into_dotdict(args.model_params))
    eval_params = load_yaml_into_dotdict(args.task_params)
    eval_params.update(load_yaml_into_dotdict(args.model_params))
    eval_params.recursive_update(load_yaml_into_dotdict('configs/eval/default.yaml'))

    # Generate 32-bit random seed, or use user-specified one
    if params.general.pytorch_and_numpy_seed is None:
        print('The seed is none')
        random_data = os.urandom(4)
        params.general.pytorch_and_numpy_seed = int.from_bytes(random_data, byteorder="big")
    print(f'Using seed: {params.general.pytorch_and_numpy_seed}')

    # Seed pytorch and numpy for reproducibility
    torch.manual_seed(params.general.pytorch_and_numpy_seed)
    torch.cuda.manual_seed_all(params.general.pytorch_and_numpy_seed)
    np.random.seed(params.general.pytorch_and_numpy_seed)

    if params.training.device == 'auto':
        params.training.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if eval_params.training.device == 'auto':
        eval_params.training.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create logger and save all code dependencies imported so far
    cur_path = os.path.dirname(os.path.abspath(__file__))
    results_folder_path = cur_path + os.sep + 'results'
    if params.arch.use_nerf:
        exp_name = 'mt3v2' + '_' +params.dataset.scenario + '_' + params.dataset.training_steps + '_' + params.dataset.task + '_' + 'lrp' + '_' + str(int(params.training.reduce_lr_patience/1000)) +'k'+'_'+'rn' + '_' + str(params.arch.multires) + '_' + params.dataset.predicts + '_' + time.strftime("%Y-%m-%d_%H%M%S")
    else:
        exp_name = 'mt3v2' + '_' +params.dataset.scenario + '_' + params.dataset.training_steps + '_' + params.dataset.task + '_' + 'lrp' + '_' + str(int(params.training.reduce_lr_patience/1000)) +'k'+ '_' + params.dataset.predicts + '_' + time.strftime("%Y-%m-%d_%H%M%S")
    logger = Logger(log_path=f'{results_folder_path}/{exp_name}', save_output=False, buffer_size=params.debug.log_interval)
    print(f"Saving results to folder {logger.log_path}")
    logger.save_code_dependencies(project_root_path=os.path.realpath('../'))  # assuming this is ran from repo root
    logger.log_scalar('seed', params.general.pytorch_and_numpy_seed, 0, flush_now=True)

    # Manually copy the configuration yaml file used for this experiment to the logger folder
    shutil.copy(args.task_params, os.path.join(logger.log_path, 'code_used', 'task_params.yaml'))
    shutil.copy(args.model_params, os.path.join(logger.log_path, 'code_used', 'model_params.yaml'))

    # Accumulate gradients to save memory
    n_splits = params.training.n_splits if params.training.n_splits is not None else 1
    if not (params.training.batch_size % n_splits == 0):
        raise ValueError("'params.training.batch_size' must be divdeble with 'params.training.n_splits'")
    params.training.batch_size = params.training.batch_size/n_splits

    model = MT3V2(params)

    # Create data generators for training
    data_generator = DataGenerator(params)

    # Create losses for training and evaluation
    mot_loss = MotLoss(params)
    mot_loss.to(params.training.device)

    #mot_loss_eval = MotLoss(eval_params)
    #mot_loss_eval.to(eval_params.training.device)

    model.to(torch.device(params.training.device))
    optimizer = AdamW(model.parameters(), lr=params.training.learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer,
                                  patience=params.training.reduce_lr_patience,
                                  factor=params.training.reduce_lr_factor,
                                  verbose=params.debug.print_reduce_lr_messages)

    current_lr = optimizer.param_groups[0]['lr']
    logger.log_scalar('metrics/learning_rate', current_lr, 0, flush_now=True)

    losses = []
    last_layer_losses = []
    GOSPA_total = []
    GOSPA_loc = []
    GOSPA_loc_n = []
    GOSPA_miss = []
    GOSPA_false = []
    GOSPA_step = []
#    eval_losses = []
#    eval_last_layer_losses = []
#    eval_gradient_step_record = []
    loop_counts = 0
    loop_eval_counts = 0
    plot_loss_dict = {}


    # load the training data and the evaluation data
    Data = mat73.loadmat(params.dataset.training_data)
    TEST = mat73.loadmat(params.dataset.test_data)

    print("[INFO] Training started...")
    start_time = time.time()
    time_since = time.time()

    total_training_index = np.random.permutation(int(params.training.n_gradient_steps))
    #total_eval_index = np.random.permutation(int(params.debug.eval_length))


    for i_gradient_step in range(params.training.n_gradient_steps*params.training.loops):
        logs = {}

        for i_split_step in range(n_splits):
            try:
                i_gs = total_training_index[i_gradient_step - params.training.n_gradient_steps * loop_counts]

                if (i_gradient_step + 1) % params.training.n_gradient_steps == 0:
                    total_training_index = np.random.permutation(int(params.training.n_gradient_steps))
                    loop_counts = loop_counts + 1


                training_data = []
                training_labels = []

                # make sure your data has at least 1 non-empty labels in each batch, otherwise the max() in datagenerator will cause error for empty input
                for i in range(int(params.training.batch_size)):
                    if Data['ES'][i_gs][0][i][0] is not None and Data['ES'][i_gs][0][i][0].any() :
                        if Data['ES'][i_gs][0][i][0].ndim == 1:
                            training_data.append(Data['ES'][i_gs][0][i][0].reshape(1,params.loss.vector_length))
                        else:
                            training_data.append(Data['ES'][i_gs][0][i][0])


                        if Data['GT'][i_gs][0][i][0] is not None and Data['GT'][i_gs][0][i][0].any() :

                            if Data['GT'][i_gs][0][i][0].ndim == 1:
                                training_labels.append(Data['GT'][i_gs][0][i][0].reshape(1,4))
                            else:
                                training_labels.append(Data['GT'][i_gs][0][i][0])
                        else:
                            training_labels.append([])



                batch, labels = data_generator.get_batch(training_data,training_labels)
                prediction, intermediate_predictions, encoder_prediction, aux_classifications, _ = model.forward(batch)

                loss_dict, indices = mot_loss.forward(labels, prediction, intermediate_predictions, encoder_prediction, loss_type=params.loss.type)

                total_loss = sum(loss_dict[k] for k in loss_dict.keys())
                logs = update_logs(logs, 'last_layer_losses', loss_dict[f'{params.loss.type}_logits'].item() + loss_dict[f'{params.loss.type}_state'].item())
                logs = update_logs(logs, 'total_loss', total_loss.item())

                if params.loss.return_intermediate:
                    for k, v in loss_dict.items():
                        if '_' in k or params.loss.type == 'dhn':
                            logs = update_logs(logs, k, v.item())

                total_loss.backward()

                if params.training.max_gradnorm is not None and params.training.max_gradnorm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), params.training.max_gradnorm)

            except KeyboardInterrupt:
                filename = f'checkpoint_gradient_step_{i_gradient_step}'
                folder_name = os.path.join(logger.log_path, 'checkpoints')
                save_checkpoint(folder=folder_name,
                                filename=filename,
                                model=model,
                                optimizer=optimizer,
                                scheduler=scheduler)
                print("[INFO] Exiting...")
                exit()



        losses.append(np.mean(np.array(logs['total_loss'])))
        last_layer_losses.append(np.mean(np.array(logs['last_layer_losses'])))

        if i_gradient_step % params.debug.print_interval == 0:
            cur_time = time.time()
            t = str(datetime.timedelta(seconds=round(cur_time - time_since)))
            t_tot = str(datetime.timedelta(seconds=round(cur_time - start_time)))
            print(f"Number of gradient steps: {i_gradient_step + 1} \t "
                f"Loss: {np.mean(losses[-15:]):.3f} \t "
                f"Time per step: {(cur_time-time_since)/params.debug.print_interval:.2f}s \t "
                f"Total time elapsed: {t_tot}")
            time_since = time.time()

        # Log all metrics
        for k,v in logs.items():
            logger.log_scalar(os.path.join('metrics', k), np.mean(np.array(v)), i_gradient_step)

        # Do the gradient step
        optimizer.step()
        optimizer.zero_grad()

        # Update learning rate, logging it if changed
        scheduler.step(np.mean(np.array(logs['total_loss'])))
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != current_lr:
            current_lr = new_lr
            logger.log_scalar('metrics/learning_rate', current_lr, i_gradient_step, flush_now=True)

        # Save checkpoint
        if i_gradient_step+1 == params.training.n_gradient_steps * params.training.loops:
            filename = f'checkpoint_gradient_step_{i_gradient_step}'
            folder_name = os.path.join(logger.log_path, 'checkpoints')
            save_checkpoint(folder=folder_name,
                            filename=filename,
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler)

        # Periodically evaluate model
        if params.debug.evaluate_gospa_interval is not None and \
                (i_gradient_step+1) % params.debug.evaluate_gospa_interval == 0:
            data_generator_eval = DataGenerator(eval_params)
            print("Starting periodic evaluation...")
            gospa_results = evaluator.evaluate_gospa(data_generator_eval, model, eval_params,params,TEST)

            GOSPA_total.append(gospa_results[0])
            GOSPA_loc.append(gospa_results[1])
            GOSPA_loc_n.append(gospa_results[2])
            GOSPA_miss.append(gospa_results[3])
            GOSPA_false.append(gospa_results[4])
            GOSPA_step.append(i_gradient_step)
            print("Done. Resuming training.")

#        if params.debug.eval_model is not None and (i_gradient_step+1) % params.debug.eval_model == 0:


#            data_generator_eval = DataGenerator(eval_params)
#            temp_index = (i_gradient_step + 1 - params.training.n_gradient_steps * loop_eval_counts) / params.debug.eval_model - 1
#            i_ev = total_eval_index[ int(temp_index) ]

#            if (i_gradient_step + 1) % params.training.n_gradient_steps == 0:
#                total_eval_index = np.random.permutation(int(params.debug.eval_length))
#                loop_eval_counts = loop_eval_counts + 1


#            eval_data = []
#            eval_labels = []

            # make sure your data has at least 1 non-empty labels in each batch, otherwise the max() in datagenerator will cause error for empty input
#            for i in range(int(params.training.batch_size)):
#                if EVAL['ES'][i_ev][0][i][0] is not None and EVAL['ES'][i_ev][0][i][0].any() :
#                    if EVAL['ES'][i_ev][0][i][0].ndim == 1:
#                        eval_data.append(EVAL['ES'][i_ev][0][i][0].reshape(1,params.loss.vector_length))
#                    else:
#                        eval_data.append(EVAL['ES'][i_ev][0][i][0])

#                    if EVAL['GT'][i_ev][0][i][0] is not None and EVAL['GT'][i_ev][0][i][0].any() :

#                        if EVAL['GT'][i_ev][0][i][0].ndim == 1:
#                            eval_labels.append(EVAL['GT'][i_ev][0][i][0].reshape(1,4))

#                        else:
#                            eval_labels.append(EVAL['GT'][i_ev][0][i][0])

#                    else:
#                        eval_labels.append([])

#            eval_loss_dict = evaluator.evaluate_model(data_generator_eval,
#                                                     eval_data,
#                                                     eval_labels,
#                                                     model,
#                                                     mot_loss_eval)

#            eval_gradient_step_record.append(i_gradient_step)
#            eval_total_loss = sum(eval_loss_dict[k] for k in eval_loss_dict.keys())
#            logs = update_logs(logs, 'eval_last_layer_losses', eval_loss_dict[f'{params.loss.type}_logits'].item() + eval_loss_dict[f'{params.loss.type}_state'].item())
#            logs = update_logs(logs, 'eval_total_loss', eval_total_loss.item())

#            if params.loss.return_intermediate:
#                for k, v in eval_loss_dict.items():
#                    if '_' in k or params.loss.type == 'dhn':
#                        logs = update_logs(logs, k, v.item())

#            eval_losses.append(np.mean(np.array(logs['eval_total_loss'])))
#            eval_last_layer_losses.append(np.mean(np.array(logs['eval_last_layer_losses'])))
#            print('The evaluation is done, resume training...')
#    print('The training is over, Saving loss data...')
    #TRaining is over saving GOSPA and loss information
    plot_loss_dict['training_total_losses'] = losses
    plot_loss_dict['training_losses'] = last_layer_losses
    plot_loss_dict['eval_GOSPA_total'] = GOSPA_total
    plot_loss_dict['eval_GOSPA_loc'] = GOSPA_loc
    plot_loss_dict['eval_GOSPA_loc_n'] = GOSPA_loc_n
    plot_loss_dict['eval_GOSPA_miss'] = GOSPA_miss
    plot_loss_dict['eval_GOSPA_false'] = GOSPA_false
    plot_loss_dict['eval_GOSPA_step'] = GOSPA_step
    print('The training is over, Saving loss data...')
    savemat('eval_loss_gospa_' + exp_name +'.mat',plot_loss_dict)
