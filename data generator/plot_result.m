function plot_result(i, ...
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

%% visualize senario setup

figure(i)
grid off;

hold on
for sensor = 1:numSensors
    theta = linspace(polarSensorPosition(sensor)-fanBearingSize/2+pi,polarSensorPosition(sensor)+fanBearingSize/2+pi);
    x = fanRangeSize*cos(theta)+cartesianSensorPosition(1,sensor);
    y = fanRangeSize*sin(theta)+cartesianSensorPosition(2,sensor);
     patch(x,y,'white','FaceColor','blue','FaceAlpha',.1,'EdgeAlpha',0);
    patch([cartesianSensorPosition(1,sensor) x(1) x(end)], ...
          [cartesianSensorPosition(2,sensor) y(1) y(end)], ...
                                                        'white', ...
                                                        'FaceColor', ...
                                                        'blue', ...
                                                        'FaceAlpha', ...
                                                        .1, ...
                                                        'EdgeAlpha', ...
                                                         0);

end
xlim([0,5])
ylim([1.2,3.5])

title(['GroundTruth,SOTA and Predictions for index:', num2str(eval_index)])
xlabel('x[m]')
ylabel('y[m]')
axis equal

% %% visualize ground truth
% hold on
% 
% numTar = length(groundTruth);
% for target = 1:numTar
%     if groundTruth(target).startTime == 1
%         scatter(groundTruth(target).state(1,1),groundTruth(target).state(2,1),50,"black");
%     else
%         scatter(groundTruth(target).state(1,1),groundTruth(target).state(2,1),50,"blue");
%     end
% 
%     hold on
% 
%     scatter(groundTruth(target).state(1,:), ...
%         groundTruth(target).state(2,:), ...
%         25,"yellow");
%     hold on
%    
% end



%% visualize Estimate

hold on
for sensor = 1:numSensors
    for track = 1:length(trajectoryEstimate{sensor}.startTime)
        states = reshape(trajectoryEstimate{sensor}.state{track},4,trajectoryEstimate{sensor}.length(track));
        scatter(states(1,:),states(2,:),'yellow','x');

        hold on
    end
end


%% visualize SOTA

for i = 1:length(BernoulliEstimates)    
    hold on;
    scatter(BernoulliEstimates(i).mean(1),BernoulliEstimates(i).mean(2),140,"k",'x');
    
end


%% visualize result from MT3
if ~isempty(MT3)
scatter(MT3(:,1),MT3(:,2),140,"b",'+');

end

%% visualize labels
hold on;
if ~isempty(labels)
scatter(labels(:,1),labels(:,2),50,"red");

end

%% Manage legend


end