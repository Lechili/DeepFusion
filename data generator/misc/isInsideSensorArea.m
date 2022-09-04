function isInside = isInsideSensorArea(targetPosition,polarSensorPosition,cartesianSensorPosition,fanBearingSize,fanRangeSize)
% check if target is in the sensor area of interest
    isInside = false;
    numSensors = length(polarSensorPosition);
    for sensor = 1:numSensors
        if norm(targetPosition-cartesianSensorPosition(:,sensor)) <= fanRangeSize
            bearing = atan2(targetPosition(2)-cartesianSensorPosition(2,sensor),targetPosition(1)-cartesianSensorPosition(1,sensor));
            if (bearing >= polarSensorPosition(sensor)-fanBearingSize/2+pi && bearing <= polarSensorPosition(sensor)+fanBearingSize/2+pi) ...
                    || (bearing >= polarSensorPosition(sensor)-fanBearingSize/2 && bearing <= polarSensorPosition(sensor)+fanBearingSize/2) ...
                    || (bearing >= polarSensorPosition(sensor)-fanBearingSize/2-pi && bearing <= polarSensorPosition(sensor)+fanBearingSize/2-pi)
                isInside = true;
            end
        end
    end
end