function [modelParameters] = positionEstimatorTraining(training_data)
      % Arguments:
      
      % - training_data:
      %     training_data(n,k)              (n = trial id,  k = reaching angle)
      %     training_data(n,k).trialId      unique number of the trial
      %     training_data(n,k).spikes(i,t)  (i = neuron id, t = time)
      %     training_data(n,k).handPos(d,t) (d = dimension [1-3], t = time)
      
      % ... train your model
      
      % Return Value:
      
      % - modelParameters:
      %     single structure containing all the learned parameters of your
      %     model and which can be used by the "positionEstimator" function.
    % Extract features and labels from training_data
    [X1_train, labels] = extractFeaturesAndLabels(training_data);  % Implement this function to extract features

    % PCA for dimensionality reduction
    [coeff, X1_train_PCA] = pca(X1_train, 'NumComponents', 20);

    % Train LDA model
    ldaModel = ldaFromScratch(X1_train_PCA, labels); % Implement ldaFromScratch
    X1_train_LDA = ldaTransform(X1_train_PCA, ldaModel); % Implement ldaTransform

    % Train Decision Trees
    treeModel = fitctree(X1_train_LDA, labels);

    % Optionally, train a Decision Tree from scratch
    maxDepth = 20;
    treeModel_scratch = buildDecisionTree(X1_train_LDA, labels, maxDepth); % Implement buildDecisionTree

    avgHandPos = calculateAverageHandPos(training_data); % Implement this function
    
    % Store models and averages in a structure
    modelParameters.pcaCoeff = coeff;
    modelParameters.ldaModel = ldaModel;
    modelParameters.treeModel = treeModel;
    modelParameters.treeModel_scratch = treeModel_scratch;
    modelParameters.avgHandPos = avgHandPos;
    modelParameters.trainMean = mean(X1_train, 1);  % mean across all trials
    % Return the model parameters
    return;
end

function [avgX, avgY] = calculateAverageHandPos(training_data)
    num_angles = 8;
    num_trials = size(training_data, 1);
    max_time_steps = 0;

    % Find the maximum time steps
    for angle = 1:num_angles
        for tr = 1:num_trials
            trial_length = size(training_data(tr, angle).handPos, 2);
            if trial_length > max_time_steps
                max_time_steps = trial_length;
            end
        end
    end

    % Initialize arrays to store the sum and count for averages
    sumX = zeros(num_angles, max_time_steps);
    sumY = zeros(num_angles, max_time_steps);
    count = zeros(num_angles, max_time_steps);

    % Sum up the x and y coordinates for each angle and time step
    for angle = 1:num_angles
        for tr = 1:num_trials
            for time = 1:size(training_data(tr, angle).handPos, 2)
                sumX(angle, time) = sumX(angle, time) + training_data(tr, angle).handPos(1, time);
                sumY(angle, time) = sumY(angle, time) + training_data(tr, angle).handPos(2, time);
                count(angle, time) = count(angle, time) + 1;
            end
        end
    end

    % Calculate the averages
    avgX = sumX ./ count;
    avgY = sumY ./ count;

    % Replace NaN values in averages (if count is zero)
    avgX(isnan(avgX)) = 0;
    avgY(isnan(avgY)) = 0;
end