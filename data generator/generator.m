function [PPP,groundTruth,trajectoryEstimate, BernoulliEstimates] = generator(numSensors,...
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
                                                                                numTarget, ...
                                                                                T_pruning, ...
                                                                                T_pruningPois, ...
                                                                                Nhyp_max, ...
                                                                                gating_threshold, ...
                                                                                existence_threshold, ...
                                                                                T_alive , ...
                                                                                existence_estimation_threshold1, ...
                                                                                H, ...                                         
                                                                                measurementNoiseCov , ...
                                                                                detectionProbability, ...
                                                                                PoissonClutterRate)


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
            if isInsideSensorArea(groundTruth(target).state(1:2,end),polarSensorPosition,cartesianSensorPosition,fanBearingSize,fanRangeSize)
                if rand <= survivalProbability
                    nextState = mvnrnd((transitionMatrix*groundTruth(target).state(:,end))',motionProcessNoiseCov)';
                    if isInsideSensorArea(nextState(1:2),polarSensorPosition,cartesianSensorPosition,fanBearingSize,fanRangeSize)
                        groundTruth(target).length = groundTruth(target).length+1;
                        groundTruth(target).state = [groundTruth(target).state nextState];
                    end
                end
            end
        end
    end
end


% multi-sensor measurement model
% each sensor may have different parameter settings
multiSensor = repmat(struct('areaOfInterest',[],'detectionProbability',[],'measurementNoiseCov',[],'PoissonClutterRate',[],'PoissonClutterIntensity',[]),[numSensors,1]);
for sensor = 1:numSensors
    multiSensor(sensor).areaOfInterest = pi*fanRangeSize^2*fanBearingSize/2*pi;
    multiSensor(sensor).measurementNoiseCov = measurementNoiseCov;
    multiSensor(sensor).detectionProbability = detectionProbability;
    multiSensor(sensor).PoissonClutterRate = PoissonClutterRate;
    multiSensor(sensor).PoissonClutterIntensity = multiSensor(sensor).PoissonClutterRate/multiSensor(sensor).areaOfInterest;
end

% generate measurements
multiSensorMeasurements = cell(numSensors);
for sensor = 1:numSensors
    multiSensorMeasurements{sensor} = cell(numSteps,1);
    for step = 1:numSteps
        multiSensorMeasurements{sensor}{step} = zeros(2,0);
    end
    for target = 1:numTarget
        for step = 1:groundTruth(target).length
            if isInsideSensorArea(groundTruth(target).state(1:2,step),polarSensorPosition(sensor),cartesianSensorPosition(:,sensor),fanBearingSize,fanRangeSize)
                if rand < multiSensor(sensor).detectionProbability
                    randMeasurements = mvnrnd(groundTruth(target).state(1:2,step)',multiSensor(sensor).measurementNoiseCov)';
                    multiSensorMeasurements{sensor}{groundTruth(target).startTime+step-1} = [multiSensorMeasurements{sensor}{groundTruth(target).startTime+step-1} randMeasurements];
                end
            end
        end
    end
    for step = 1:numSteps
        % add clutter measurements
        numClutter = poissrnd(multiSensor(sensor).PoissonClutterRate);
        for clutter = 1:numClutter
            uniformRandRange = fanRangeSize*rand;
            uniformRandBearing = (2*rand-1)*fanBearingSize/2+polarSensorPosition(sensor)+pi;
            randClutter = uniformRandRange*[cos(uniformRandBearing);sin(uniformRandBearing)];
            multiSensorMeasurements{sensor}{step} = [multiSensorMeasurements{sensor}{step} randClutter+cartesianSensorPosition(:,sensor)];
        end
    end
end



% wrapper for running a multi-object tracker
parameters.PoissonBirthRate = PoissonBirthRate;
parameters.PoissonBirthGaussianMean = PoissonBirthGaussianMean;
parameters.PoissonBirthGaussianCov = PoissonBirthGaussianCov;
parameters.survivalProbability = survivalProbability;
parameters.transitionMatrix = transitionMatrix;
parameters.motionProcessNoiseCov = motionProcessNoiseCov;
parameters.numInitialTarget = numInitialTarget;

trajectoryEstimate = cell(numSensors,1);
marginalPMB = cell(numSensors,1);
for sensor = 1:numSensors
    parameters.measurementNoiseCov = multiSensor(sensor).measurementNoiseCov;
    parameters.detectionProbability = multiSensor(sensor).detectionProbability;
    parameters.PoissonClutterIntensity = multiSensor(sensor).PoissonClutterIntensity;
    [trajectoryEstimate{sensor},marginalPMB{sensor}] = wrapperMultiObjectTracker(multiSensorMeasurements{sensor},parameters, ...
                                                                                  T_pruning, ...
                                                                                  T_pruningPois, ...
                                                                                  Nhyp_max, ...
                                                                                  gating_threshold, ...
                                                                                  existence_threshold, ...
                                                                                  T_alive , ...
                                                                                  existence_estimation_threshold1, ...
                                                                                  H);
end



%% Decentralized fusion

for sensor = 1:numSensors
    numPoissonComponent = length(marginalPMB{sensor}.Poisson);
    if numPoissonComponent > 1
        weights = [marginalPMB{sensor}.Poisson.weight];
        mergedWeight = sum(weights);
        mergedMean = [marginalPMB{sensor}.Poisson.mean]*weights';
        mergedCovariance = zeros(4);
        for i = 1:numPoissonComponent
            x_diff = marginalPMB{sensor}.Poisson(i).mean - mergedMean;
            mergedCovariance = mergedCovariance + weights(i).*(marginalPMB{sensor}.Poisson(i).covariance + x_diff*x_diff');
        end
        marginalPMB{sensor}.Poisson(2:end) = [];
        marginalPMB{sensor}.Poisson.weight = mergedWeight;
        marginalPMB{sensor}.Poisson.mean = mergedMean;
        marginalPMB{sensor}.Poisson.covariance = mergedCovariance;
    end
end

Ns = cellfun(@(x) length(x.multiBernoulli),marginalPMB);
K = sum(Ns);
Ms = K-Ns;

multiBernoulli = cell(numSensors,1);
for sensor = 1:numSensors
    multiBernoulli{sensor} = marginalPMB{sensor}.multiBernoulli;
    for i = 1:Ms(sensor)
        multiBernoulli{sensor}(Ns(sensor)+i).weight = marginalPMB{sensor}.Poisson.weight/Ms(sensor);
        multiBernoulli{sensor}(Ns(sensor)+i).mean = marginalPMB{sensor}.Poisson.mean;
        multiBernoulli{sensor}(Ns(sensor)+i).covariance = marginalPMB{sensor}.Poisson.covariance;
    end
end

assignments = zeros(numSensors,K);
assignments(1,:) = 1:K;
for sensor = 2:numSensors
    costMatrix = zeros(K);
    for i = 1:K
        for j = 1:K
            costMatrix(i,j) = BernoulliKLD(multiBernoulli{1}(i),multiBernoulli{sensor}(j));
        end
    end
    % non-negative assign 0 to negative value
    [C_R, C_C] = size(costMatrix);
    for i = 1 : C_R
        for j = 1:C_C
            if costMatrix(i,j) < 0
                costMatrix(i,j) = 0;
            end
        end
    end
    [assignments(sensor,:),~] = assignmentoptimal(costMatrix);
end

assignmentsPorB = false(numSensors,K);
for i = 1:numSensors
    assignmentsPorB(i,assignments(i,:)<=Ns(i)) = true;
end

Bernoulli = [];
PPP = [];
for i = 1:K
    if sum(assignmentsPorB(:,i)) == 3
        BernoulliMixture = [];
        for sensor = 1:numSensors
            BernoulliMixture = [BernoulliMixture multiBernoulli{sensor}(assignments(sensor,i))];
        end
        Bernoulli = [Bernoulli BernoulliFusion(BernoulliMixture)];
    elseif sum(assignmentsPorB(:,i)) == 0
        PoissonPointProcessMixture = [];
        for sensor = 1:numSensors
            PoissonPointProcessMixture = [PoissonPointProcessMixture multiBernoulli{sensor}(assignments(sensor,i))];
        end
        PoissonPointProcess = PoissonFusion(PoissonPointProcessMixture);
        PPP = [PPP PoissonPointProcess];
    else
        BernoulliMixture = [];
        PoissonPointProcessMixture = [];
        for sensor = 1:numSensors
            if assignmentsPorB(sensor,i)
                BernoulliMixture = [BernoulliMixture multiBernoulli{sensor}(assignments(sensor,i))];
            else
                PoissonPointProcessMixture = [PoissonPointProcessMixture multiBernoulli{sensor}(assignments(sensor,i))];
            end
        end
        Bernoulli = [Bernoulli BernoulliPoissonFusion(BernoulliMixture,PoissonPointProcessMixture)];
    end
end

% extract estimates at last time step
BernoulliEstimates =  Bernoulli;


end
