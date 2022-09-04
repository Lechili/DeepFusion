% To collect data for scenario 3, where sensors are mobile.

clear;
clc;

addpath Assignment/
addpath 'GOSPA code'/
addpath 'TPMBM filter'/
addpath 'misc'

% Enter the file path, and the file name(should end with .mat)

where_to_save = '';
file_name = '';

NumSamples = 1000;
batch_size = 1;

% number of sensors
numSensors = 3;

% each sensor has fan-shaped area of interest
fanBearingSize = 2*pi/3;
fanRangeSize = 20;


% sensors are located on a line
circleRadius = 10;
polarSensorPosition = -pi/2 ;
polarSensorPosition1 = pi/2 ;

polarSensorPosition = [-pi/2 -pi/2 pi/2];

PoissonBirthRate = .1;
PoissonBirthGaussianMean = [0;0;0;0];
PoissonBirthGaussianCov = diag([10 10 5 5]);
survivalProbability = .95;
scanTime = .1;
% nearly constant velocity motion model [2D position;2D velocity]

transitionMatrix = kron([1 scanTime;0 1],eye(2));
accelerationDeviation = .5;
motionProcessNoiseCov = accelerationDeviation^2*kron([scanTime^3/3 scanTime^2/2;scanTime^2/2 scanTime],eye(2));
sensor_accelerationDeviation = 100;
sensor_motionProcessNoiseCov = sensor_accelerationDeviation*kron([scanTime^3/3 scanTime^2/2;scanTime^2/2 scanTime],eye(2));
% generate ground truth
numSteps = 10;
numInitialTarget = 3;
numTarget = 0;

SOTA = cell(NumSamples,1);
ES = cell(NumSamples,1);
GT = cell(NumSamples,1);
org_ES = cell(NumSamples,1);
org_GT = cell(NumSamples,1);
PPP = cell(NumSamples,1);
PS = cell(NumSamples,1);



parfor i = 1:NumSamples
    [ps,ppp,gt,es,training_data,labels,sota] = processor_mobile(i, ...
                                                            batch_size, ...
                                                            numSensors,...
                                                            fanBearingSize, ...
                                                            fanRangeSize, ...
                                                            polarSensorPosition, ... 
                                                            PoissonBirthRate, ... 
                                                            PoissonBirthGaussianMean, ... 
                                                            PoissonBirthGaussianCov, ... 
                                                            survivalProbability, ... 
                                                            transitionMatrix, ...
                                                            motionProcessNoiseCov, ...
                                                            numSteps, ...
                                                            numInitialTarget, ...
                                                            numTarget, ...
                                                            sensor_motionProcessNoiseCov);
    ES{i} = training_data;
    GT{i} = labels;
    SOTA{i} = sota;
    org_ES{i} = es;
    org_GT{i} = gt;
    PPP{i} = ppp;
    PS{i} = ps;
end

%%%
f = fullfile(where_to_save,file_name);
save(f,'PS','PPP','ES','GT','SOTA','org_ES','org_GT','-v7.3');
