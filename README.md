# Introduction
This repository contains the code for the paper "DEEP FUSION OF MULTI-OBJECT DENSITIES USING TRANSFORMER". The code was developed as joint effort by Lechi Li, Chen Dai, Yuxuan Xia and Lennart Svensson, and was based on the code available at the repositories for [TPMBM](https://github.com/Agarciafernandez/MTT/tree/master/TPMBM%20filter) and [MT3v2](https://github.com/JulianoLagana/MT3v2).
# Setting up
The MATLAB version used in this project is MATLAB R2021b. Follow the instruction in [MT3v2](https://github.com/JulianoLagana/MT3v2) to set up a conda environment. After activating the environment, run the command:

```
pip install mat73
```
# Data generation
The data used in this work are synthetic data, which is generated in MATLAB, by using ```collecting_data.m``` ( for scenario 1 and 2 ) and ```collecting_data_mobile.m``` ( for scenario 3 ) under ```data generator```. One can also specify hyperparameters such as the number of sensors, sensor noise level, filtering parameters, etc, in these files. Note that one should specify the path and name of the dataset in these files by hand before collecting.

# Training
Training hyperparameters such as batch size, learning rate, checkpoint interval, etc, are found in the file ```configs/models/mt3v2.yaml```. Note that one should specify the path and name of the dataset in ```mt3v2.yaml``` by hand before training.

To train the model, run the command:

```
src/training.py -tp configs/tasks/task1.yaml -mp configs/models/mt3v2.yaml
```

# Evaluation 
Firstly, run the following command:
```
src/test.py -rp src/results/experiment_name -tp configs/tasks/task1.yaml
```
After running the above command it will generate a '.mat' file. Copy this file under the folder ```data generator```, and use ```data generator/test_mt3v2.m``` to evluate the GOSPA score.

To evaluate the NLL, tune the PPP component first using ```src/ppp_tuner_for_mt3.py```, by running the following command:
```
src/ppp_tuner_for_mt3.py -rp src/results/experiment_name -tp configs/tasks/task1.yaml
```
After tuning PPP, use the tuned PPP intensities to evluate NLL in ```src/test_nll.py```. Run the command: 
```
src/test_nll.py -rp src/results/experiment_name -tp configs/tasks/task1.yaml
```
