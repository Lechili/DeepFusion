function [sensor_positions,PPP,groundTruth,trajectoryEstimate, BernoulliEstimates] = generator_mobile(numSensors,...
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
                                                                                                sensor_motionProcessNoiseCov)


sensor_positions = cell(numSensors,1);
for i=1:numSensors
    if i==3
        sensor_positions{i}=zeros(4,numSteps);
        sensor_positions{i}(:,1)=[0;10;0;16];
        for j=2:numSteps
            sensor_positions{i}(:,j)=mvnrnd(transitionMatrix*sensor_positions{i}(:,j-1),sensor_motionProcessNoiseCov)';
        end
    end
    
    if i==2
        sensor_positions{i}=zeros(4,numSteps);
        sensor_positions{i}(:,1)=[0.5;-10;15;0];
        for j=2:numSteps
            sensor_positions{i}(:,j)=mvnrnd(transitionMatrix*sensor_positions{i}(:,j-1),sensor_motionProcessNoiseCov)';
        end
    end
    
    if i==1
        sensor_positions{i}=zeros(4,numSteps);
        sensor_positions{i}(:,1)=[-0.5;-10;-15;0];
        for j=2:numSteps
            sensor_positions{i}(:,j)=mvnrnd(transitionMatrix*sensor_positions{i}(:,j-1),sensor_motionProcessNoiseCov)';
        end
    end        
end



for step = 1:numSteps
    
    % don't add new objects in the last two timesteps of generation, for cleaner training labels
    if step < 9
         numNewBorn = poissrnd(PoissonBirthRate);
    else
         numNewBorn = 0;
    end

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
            SensorPosition = [sensor_positions{1,1}(1:2,step) sensor_positions{2,1}(1:2,step) sensor_positions{3,1}(1:2,step)]; 
            if isInsideSensorArea(groundTruth(target).state(1:2,end),polarSensorPosition,SensorPosition,fanBearingSize,fanRangeSize)
                if rand <= survivalProbability
                    nextState = mvnrnd((transitionMatrix*groundTruth(target).state(:,end))',motionProcessNoiseCov)';
                    SensorPosition_next = [sensor_positions{1,1}(1:2,step+1) sensor_positions{2,1}(1:2,step+1) sensor_positions{3,1}(1:2,step+1)];
                    if isInsideSensorArea(nextState(1:2),polarSensorPosition,SensorPosition_next,fanBearingSize,fanRangeSize)
                        groundTruth(target).length = groundTruth(target).length+1;
                        groundTruth(target).state = [groundTruth(target).state nextState];

                    end
                end
            end
        end
    end

end

% 
% 
% multi-sensor measurement model
% each sensor may have different parameter settings
multiSensor = repmat(struct('areaOfInterest',[],'detectionProbability',[],'measurementNoiseCov',[],'PoissonClutterRate',[],'PoissonClutterIntensity',[]),[numSensors,1]);
for sensor = 1:numSensors
    multiSensor(sensor).areaOfInterest = pi*fanRangeSize^2*fanBearingSize/2*pi;
    multiSensor(sensor).measurementNoiseCov = eye(2)*0.01;   %0.01
    multiSensor(sensor).detectionProbability = .9;
    multiSensor(sensor).PoissonClutterRate = 5;
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
            global_step = step+groundTruth(target).startTime-1;    %set a global step as step not always starts with 1
            Sensor_Position = sensor_positions{sensor,1}(1:2,global_step);
            if isInsideSensorArea(groundTruth(target).state(1:2,step),polarSensorPosition(sensor),Sensor_Position,fanBearingSize,fanRangeSize) %cartesianSensorPosition(:,sensor)
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
            multiSensorMeasurements{sensor}{step} = [multiSensorMeasurements{sensor}{step} randClutter+sensor_positions{sensor,1}(1:2,step)];
        end
    end
end


% % wrapper for running a multi-object tracker
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
    [trajectoryEstimate{sensor},marginalPMB{sensor}] = wrapperMultiObjectTracker(multiSensorMeasurements{sensor},parameters);
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

%extract estimates at last time step
BernoulliEstimates = Bernoulli;%([Bernoulli.weight]>=0.5);

end