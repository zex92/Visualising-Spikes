function [X, labels] = extractFeaturesAndLabels(training_data)
    % Initialize the feature matrix X and labels vector
    numTrials = size(training_data, 1);
    numAngles = 8;
    numNeurons = 98;  % Assuming all trials have the same number of neurons

    % Loop over each trial and angle
    for angle = 1:numAngles
        for idx = 1:numTrials
            row = (angle - 1) * numTrials + idx; % Calculate the row index in X
            for neuron = 1:numNeurons
                X(row, neuron) = sum(training_data(idx, angle).spikes(neuron, :));
            end
            labels(row) = angle; % Store the corresponding angle in labels
        end
    end
end
