# Introduction
This repository contains the code for the paper "DEEP FUSION OF MULTI-OBJECT DENSITIES USING TRANSFORMER". The code was developed as joint effort by Lechi Li, Chen Dai, Yuxuan Xia and Lennart Svensson, and was based on the code available at the repositories for [TPMBM](https://github.com/Agarciafernandez/MTT/tree/master/TPMBM%20filter) and [MT3v2](https://github.com/JulianoLagana/MT3v2).
# Setting up
In order to set up a conda environment with all the necessary dependencies, run the command:

```
conda env create -f conda-env/environment-<gpu/cpu>.yml
```
# Data generation
The data used in this work are synthetic data, which is generated in MATLAB. The training and test data are generated using collect_trainig.m and collecting_test.m, respectivly. One can also specify hyperparameters such as the number of sensors, sensor noise level, filtering parameters, etc, in these files.
# Training
Training hyperparameters such as batch size, learning rate, checkpoint interval, etc, are found in the file configs/models/mt3v2.yaml. Note that one should specify the path of the dataset in mt3v2.yaml by hand before training.

To train the model, run the command:

```
src/training.py -tp configs/tasks/task1.yaml -mp configs/models/mt3v2.yaml
```

# Evaluation 
The GOSPA score of the models are evaluated using test_MT3v2.m. To evaluate the NLL, one has to tune the PPP component using ppp_tuner_for_mt3.py and then use the tuned PPP intensities to evluate NLL test_nll.py. 
```
conda env create -f conda-env/environment-<gpu/cpu>.yml
```


To evaluate the NLL, one has to tune the PPP component first by running the following command.
```
src/ppp_tuner_for_mt3.py -rp src/results/experiment_name -tp configs/tasks/task1.yaml
```
After tuning PPP, run the command: 
```
src/test_nll.py -rp src/results/experiment_name -tp configs/tasks/task1.yaml
```
