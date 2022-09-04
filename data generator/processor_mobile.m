function [PS,PPP,GT,ES,training_data,labels,SOTA] = processor_mobile(index, ...
                                                            N_s, ...
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
                                                            sensor_motionProcessNoiseCov) 

rng(index);

GT = cell(N_s,1);
ES = cell(N_s,1);
SOTA = cell(N_s,1);
PS = cell(N_s,1);
PPP = cell(N_s,1);

for t=1:N_s

[sensor_positions,ppp,groundTruth,trajectoryEstimate,BernoulliEstimates] = generator_mobile(numSensors,...
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

    GT{t} = groundTruth ;
    ES{t} = trajectoryEstimate;
    SOTA{t} = BernoulliEstimates;
    PPP{t} = ppp;
    PS{t} = sensor_positions;
end

packed_data = cell(numSteps,N_s);
packed_ground_truth = cell(1,N_s);

% The first for loops over all batches
for a = 1:N_s
   % number of trajectories
   Num_traj = size(GT{a,1},2);

   % The second for loops over all trajectories within that batch
   for b = 1:Num_traj
        if GT{a,1}(b).startTime + GT{a,1}(b).length -1 == numSteps
               packed_ground_truth{1,a} = [packed_ground_truth{1,a} ; GT{a,1}(b).state(:,GT{a,1}(b).length)'];
        end
   end
end


% 
for i = 1:N_s % loop over batch
   for j = 1:numSensors % loop over sensors
       Num_traj = length(ES{i,1}{j,1}.startTime);
       for k = 1:Num_traj % loop over trajectories
           
           % get the existance probability
           ex_p = ES{i,1}{j,1}.existence_estimate(k);
           % get the alive probability
           alive_p = ES{i,1}{j,1}.alive_prob(k);
           % get the estimates
           temp = reshape(ES{i,1}{j,1}.state{1,k},4,length(ES{i,1}{j,1}.state{1,k})/4);
           for l = 1:ES{i,1}{j,1}.length(k) % loop over all the states
               % Take out the corresponding covariance matrix
               temp1 = ES{i,1}{j,1}.covariance{1,k}((4*l-3):l*4,(4*l-3):l*4);
               % make sure we have symmetric matrix 
               temp1 = (temp1' + temp1) /2;
               % create a mask to select the elements of the resulting upper triangle matrix
               mask = ~bsxfun(@gt,[1:size(temp1,1)]',1:size(temp1,2));
               % select the elements and transpose to a row vector
               temp1 = temp1(mask)'; 
               % place the states, covariance ,final probability , time
               % step, sensor ids, trj ids to a vector
               packed_data{ES{i,1}{j,1}.startTime(k)+l-1,i}= [packed_data{ES{i,1}{j,1}.startTime(k)+l-1,i}; [temp(:,l)' PS{i,1}{j,1}(1,ES{i,1}{j,1}.startTime(k)+l-1) PS{i,1}{j,1}(2,ES{i,1}{j,1}.startTime(k)+l-1) polarSensorPosition(j) temp1 ex_p*alive_p (ES{i,1}{j,1}.startTime(k)+l-2)/10 j k ]];
           end
       end
   end
end

labels = cell(N_s,1);
training_data = cell(N_s,1);
for n = 1:N_s
    if packed_ground_truth{1,n}
        labels{n} = packed_ground_truth{1,n}(randperm(size(packed_ground_truth{1,n}, 1)), :);
    else
        labels{n} = [];
    end
    for m = 1:numSteps
        training_data{n} = [training_data{n};packed_data{m,n}(randperm(size(packed_data{m,n}, 1)), :)];
    end
  
end

end