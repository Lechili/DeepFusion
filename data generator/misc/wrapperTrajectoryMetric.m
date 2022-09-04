function [X,Y] = wrapperTrajectoryMetric(numSteps,groundTruth,trajectoryEstimate)

nx = length(groundTruth);
X.tVec = [groundTruth.startTime]';
X.iVec = [groundTruth.length]';
X.xState = nan(4,numSteps,nx);
for track = 1:nx
    for step = X.tVec(track):X.tVec(track)+X.iVec(track)-1
        X.xState(:,step,track) = groundTruth(track).state(:,step-X.tVec(track)+1);
    end
end

ny = length(trajectoryEstimate.startTime);
Y.tVec = trajectoryEstimate.startTime';
Y.iVec = trajectoryEstimate.length';
Y.xState = nan(4,numSteps,ny);
for track = 1:ny
    states = reshape(trajectoryEstimate.state{track},4,trajectoryEstimate.length(track));
    for step = Y.tVec(track):Y.tVec(track)+Y.iVec(track)-1
        Y.xState(:,step,track) = states(:,step-Y.tVec(track)+1);
    end
end

end

