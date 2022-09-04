% This script computes the average GOSPA and NLL scores for the results from MT3v2 and
% SOTA. In addition, it also generates some number of plots for them.
% You should have your prediction MAT file and evaluation MAT file ready.

% Creater: Lechi Li

clear;
clc;

addpath 'GOSPA code'/
addpath test/

% Enter the path of MT3v2 predictions and the path of the test data
prediction_path = '';
test_path = '';
% Set GOSPA parameters
p = 1; c = 2; alpha = 2; ndim = 4;

% Existance Threshold
sota_threshold = 0.5;
mt3v2_threshold = 0.75;


%Number of sensors
numSensors = 3;

% Sensor position
fanBearingSize = 2*pi/3;
fanRangeSize = 20;
e
circleRadius = 10;
polarSensorPosition = -pi/2 ;
polarSensorPosition1 = pi/2 ;
cartesianSensorPosition1 =circleRadius*[cos(polarSensorPosition1);sin(polarSensorPosition1)];
cartesianSensorPosition = circleRadius*[cos(polarSensorPosition);sin(polarSensorPosition)];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Sensor positions for scenario 1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pos1 = cartesianSensorPosition+[-5;0];
pos2 = cartesianSensorPosition+[5;0];
pos3 =cartesianSensorPosition1+[0;7.5];
cartesianSensorPosition = [pos1 pos2 pos3];
polarSensorPosition = [-pi/2 -pi/2 pi/2];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Sensor positions for scenario 2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%pos1 = cartesianSensorPosition+[-0.5;0];
%pos2 = cartesianSensorPosition+[0.5;0];
%pos3 =cartesianSensorPosition1+[0;0];
%cartesianSensorPosition = [pos1 pos2 pos3];
%polarSensorPosition = [-pi/2 -pi/2 pi/2];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Load the predictions of MT3v2
predictions = load(prediction_path);
predictions = struct2cell(predictions);

n_eval = lenght(predictions);

% Load the test data
ValData = load(test_path);
ValData.origin_SOTA = ValData.SOTA;

% Extract estimates from MT3v2 and Bayesian method using existence
% thresholds
predictions_gospa = cell(n_eval,1);

for i = 1:n_eval
    mask1 = (predictions{i,1}(:,9) > mt3v2_threshold);
    predictions_gospa{i,1} = predictions{i,1}(mask1,:);

    mask2 = ([ValData.origin_SOTA{i,1}{1,1}.weight] > sota_threshold);
    ValData.SOTA{i,1}{1,1} = ValData.origin_SOTA{i,1}{1,1}(mask2);

end
%Compute GOSPA scores and also return some additional information for the
%evaluation dataset.
[add_info,MT3_GOSPA,SOTA_GOSPA] = avg_gospa_score(n_eval, ...
                                                  predictions_gospa, ...
                                                  ValData, ...
                                                  p, ...
                                                  c, ...
                                                  alpha, ...
                                                  ndim);

% Display the relevant information 
MT3_gospa = ['The average GOSPA score for MT3 is:',num2str(MT3_GOSPA.MT3_scores),'.'];
disp(MT3_gospa)

MT3_loc = ['The average localisation error for MT3 is:',num2str(MT3_GOSPA.MT3_loc_errors),'.'];
disp(MT3_loc)

MT3_missed = ['The average missed error for MT3 is:',num2str(MT3_GOSPA.MT3_missed_errors),'.'];
disp(MT3_missed)

MT3_false = ['The average false error for MT3 is:',num2str(MT3_GOSPA.MT3_false_errors),'.'];
disp(MT3_false)

SOTA_gospa = ['The average GOSPA score for SOTA is:',num2str(SOTA_GOSPA.SOTA_scores),'.'];
disp(SOTA_gospa)

SOTA_loc = ['The average localisation error for SOTA is:',num2str(SOTA_GOSPA.SOTA_loc_errors),'.'];
disp(SOTA_loc)

SOTA_missed = ['The average missed error for SOTA is:',num2str(SOTA_GOSPA.SOTA_missed_errors),'.'];
disp(SOTA_missed)

SOTA_false = ['The average false error for SOTA is:',num2str(SOTA_GOSPA.SOTA_false_errors),'.'];
disp(SOTA_false)

disp(['-----------------------------------------------------------------------------------'])


% random generate some number of images and plot

avaliable_index = [1:n_eval];
%Choose the number of plots and generate random index for plotting
num_plots = 3;
rand_index = randsample(avaliable_index,num_plots);
for i = 1:length(rand_index)

    eval_index = rand_index(i);
    MT3 = predictions_gospa{eval_index,1};
    groundTruth = ValData.org_GT{eval_index,1}{1,1};
    trajectoryEstimate = ValData.org_ES{eval_index,1}{1,1};
    BernoulliEstimates = ValData.SOTA{eval_index,1}{1,1};
    labels = ValData.GT{eval_index,1}{1,1};

    %%visual the result
    plot_result(i, ...
                eval_index, ...
                numSensors, ...
                 polarSensorPosition, ...
                 cartesianSensorPosition, ...
                 fanBearingSize, ...
                 fanRangeSize, ...
                 trajectoryEstimate, ...             
                 groundTruth, ...
                 labels, ...
                 BernoulliEstimates, ...
                 MT3)
end
