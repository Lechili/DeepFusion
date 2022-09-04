clear;
clc;

addpath Assignment/
addpath 'TPMBM filter'/
addpath 'misc'

% Enter the file path, and the file name(should end with .mat)

where_to_save = '';
file_name = '';

% Number of samples
NumSamples = 10;

batch_size = 1;

% Use different range of seed when generating training and validation data
Se_ed = 0;

% number of sensors
numSensors = 3;

% each sensor has fan-shaped area of interest
fanBearingSize = 2*pi/3;
fanRangeSize = 20;


% sensors are located on a line
circleRadius = 10;
polarSensorPosition = -pi/2 ;
polarSensorPosition1 = pi/2 ;
cartesianSensorPosition1 =circleRadius*[cos(polarSensorPosition1);sin(polarSensorPosition1)];
cartesianSensorPosition = circleRadius*[cos(polarSensorPosition);sin(polarSensorPosition)];

% type the offset of the sensor position here
pos1 = cartesianSensorPosition+[-5;0];
pos2 = cartesianSensorPosition+[5;0];
pos3 =cartesianSensorPosition1+[0;7.5];
cartesianSensorPosition = [pos1 pos2 pos3];
polarSensorPosition = [-pi/2 -pi/2 pi/2];

% Model parameters
PoissonBirthRate = .1;
PoissonBirthGaussianMean = [0;0;0;0];
PoissonBirthGaussianCov = diag([10 10 5 5]);
survivalProbability = .95;
scanTime = .1;

% Nearly constant velocity motion model [2D position;2D velocity]

transitionMatrix = kron([1 scanTime;0 1],eye(2));
accelerationDeviation = .5;
motionProcessNoiseCov = accelerationDeviation^2*kron([scanTime^3/3 scanTime^2/2;scanTime^2/2 scanTime],eye(2));

%Measurement parameters
measurementNoiseCov = eye(2)*0.01;
detectionProbability = .9;
PoissonClutterRate = 5;

% Generate ground truth
numSteps = 10;
numInitialTarget = 3;
numTarget = 0;

% Filtering parameters (local tracker)
T_pruning = 0.001;
T_pruningPois = 0.00001;
Nhyp_max = 100;
gating_threshold = 20;
existence_threshold = 0.001;
T_alive = 0.0001;
existence_estimation_threshold1 = 0.5; %lower this parameter to extract more estimates, choose something between 0 and 0.5
H = [1 0 0 0;0 1 0 0];


% Prepare room for results from the Bayesian method
SOTA = cell(NumSamples,1);

% Prepare room for trajectory estimates (after transformation)
ES = cell(NumSamples,1);

% Prepare room for ground truth(after transformation)
GT = cell(NumSamples,1);

% Prepare room for trajectory estimates (before transformation)
org_ES = cell(NumSamples,1);

% Prepare room for trajectory estimates (before transformation)
org_GT = cell(NumSamples,1);

% Prepare room for trajectory estimates (before transformation)
PPP = cell(NumSamples,1);


for i = 1:NumSamples % parfor

    [ppp,gt,es,training_data,labels,sota] = processor(i+Se_ed, ...
                                            batch_size, ...
                                            numSensors,...
                                            fanBearingSize, ...
                                            fanRangeSize, ...
                                            polarSensorPosition, ... 
                                            cartesianSensorPosition, ...
                                            PoissonBirthRate, ... 
                                            PoissonBirthGaussianMean, ... 
                                            PoissonBirthGaussianCov, ... 
                                            survivalProbability, ... 
                                            transitionMatrix, ...
                                            motionProcessNoiseCov, ...
                                            numSteps, ...
                                            numInitialTarget, ...
                                            numTarget, ...                                                                                T_pruning, ...
                                            T_pruningPois, ...
                                            Nhyp_max, ...
                                            gating_threshold, ...
                                            existence_threshold, ...
                                            T_alive , ...
                                            existence_estimation_threshold1, ...
                                            H, ...
                                            measurementNoiseCov , ...
                                            detectionProbability, ...
                                            PoissonClutterRate);

    ES{i} = training_data;
    GT{i} = labels;
    SOTA{i} = sota;
    org_ES{i} = es;
    org_GT{i} = gt;
    PPP{i} = ppp;
end

% save the file 
f = fullfile(where_to_save,file_name);
save(f,'PPP','ES','GT','SOTA','org_ES','org_GT','-v7.3');