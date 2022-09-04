clear;close all;clc
dbstop if error
% rng default

% region of interest is a circle, centred at origin
cartesianSensorPosition = [0;0];
circleRadius = 200;

% multi-object motion model
PoissonBirthRate = .1;
PoissonBirthGaussianMean = zeros(4,1);
PoissonBirthGaussianCov = diag([20 20 2 2].^2);
survivalProbability = .98;
% nearly constant velocity motion model [2D position;2D velocity]
scanTime = 1;
transitionMatrix = kron([1 scanTime;0 1],eye(2));
accelerationDeviation = 1;
motionProcessNoiseCov = accelerationDeviation^2*kron([scanTime^3/3 scanTime^2/2;scanTime^2/2 scanTime],eye(2));

% generate ground truth
numSteps = 100;
numInitialTarget = 3;
numTarget = 0;
for step = 1:numSteps
    numNewBorn = poissrnd(PoissonBirthRate);
    if step == 1
        numNewBorn = numNewBorn+numInitialTarget;
    end

    for target = 1:numNewBorn
        numTarget = numTarget+1;
        groundTruth(numTarget).startTime = step;
        groundTruth(numTarget).length = 1;
        groundTruth(numTarget).state = mvnrnd(PoissonBirthGaussianMean',PoissonBirthGaussianCov)';
    end

    for target = 1:numTarget
        if step~=numSteps && groundTruth(target).startTime+groundTruth(target).length-1==step
            if norm(groundTruth(target).state(1:2,end)-cartesianSensorPosition) <= circleRadius
                if rand <= survivalProbability
                    nextState = mvnrnd((transitionMatrix*groundTruth(target).state(:,end))',motionProcessNoiseCov)';
                    if norm(nextState(1:2)-cartesianSensorPosition) <= circleRadius
                        groundTruth(target).length = groundTruth(target).length+1;
                        groundTruth(target).state = [groundTruth(target).state nextState];
                    end
                end
            end
        end
    end
end

% visualize ground truth
figure
hold on

theta = linspace(-pi,pi);
x = circleRadius*cos(theta);
y = circleRadius*sin(theta);
patch(x,y,'white','FaceColor','blue','FaceAlpha',.1,'EdgeAlpha',0)

for target = 1:numTarget
    scatter(groundTruth(target).state(1,1),groundTruth(target).state(2,1),50,'k','LineWidth',2)
    plot(groundTruth(target).state(1,:),groundTruth(target).state(2,:),'k','LineWidth',2)
end
xlim([-200,200])
ylim([-200,200])
axis equal

% multi-object measurement model
areaOfInterest = pi*circleRadius^2;
measurementNoiseCov = eye(2);
detectionProbability = .95;
PoissonClutterRate = 10;
PoissonClutterIntensity = PoissonClutterRate/areaOfInterest;

% generate measurements
sensorMeasurements = cell(numSteps,1);
for step = 1:numSteps
    sensorMeasurements{step} = zeros(2,0);
end
for target = 1:numTarget
    for step = 1:groundTruth(target).length
        if norm(groundTruth(target).state(1:2,end)-cartesianSensorPosition) <= circleRadius
            if rand < detectionProbability
                randMeasurements = mvnrnd(groundTruth(target).state(1:2,step)',measurementNoiseCov)';
                sensorMeasurements{groundTruth(target).startTime+step-1} = [sensorMeasurements{groundTruth(target).startTime+step-1} randMeasurements];
            end
        end
    end
end
for step = 1:numSteps
    % add clutter measurements
    numClutter = poissrnd(PoissonClutterRate);
    for clutter = 1:numClutter
        uniformRandRange = circleRadius*rand;
        uniformRandBearing = (2*rand-1)*pi;
        randClutter = uniformRandRange*[cos(uniformRandBearing);sin(uniformRandBearing)];
        sensorMeasurements{step} = [sensorMeasurements{step} randClutter+cartesianSensorPosition];
    end
end

% visualize measurement
% for step = 1:numSteps
%     if ~isempty(sensorMeasurements{step})
%         scatter(sensorMeasurements{step}(1,:),sensorMeasurements{step}(2,:),20,'k','filled')
%     end
% end

% wrapper for running a multi-object tracker
parameters.PoissonBirthRate = PoissonBirthRate;
parameters.PoissonBirthGaussianMean = PoissonBirthGaussianMean;
parameters.PoissonBirthGaussianCov = PoissonBirthGaussianCov;
parameters.survivalProbability = survivalProbability;
parameters.transitionMatrix = transitionMatrix;
parameters.motionProcessNoiseCov = motionProcessNoiseCov;
parameters.numInitialTarget = numInitialTarget;
parameters.measurementNoiseCov = measurementNoiseCov;
parameters.detectionProbability = detectionProbability;
parameters.PoissonClutterIntensity = PoissonClutterIntensity;

% multi-object trajectory estimation
trajectoryEstimate = wrapperMultiObjectTracker(sensorMeasurements,parameters);

% performance evaluation using LP trajectory metric
[X,Y] = wrapperTrajectoryMetric(numSteps,groundTruth,trajectoryEstimate);
% parameters of LP trajectory metric
c = 20;
p = 1;
gamma = 2;
[dxy,loc_cost,miss_cost,fa_cost,switch_cost] = LPTrajMetric_cluster(X,Y,c,p,gamma);
