function [x, y] = positionEstimator(test_data, modelParameters)

  % **********************************************************
  %
  % You can also use the following function header to keep your state
  % from the last iteration
  %
  % function [x, y, newModelParameters] = positionEstimator(test_data, modelParameters)
  %                 ^^^^^^^^^^^^^^^^^^
  % Please note that this is optional. You can still use the old function
  % declaration without returning new model parameters. 
  %
  % *********************************************************

  % - test_data:
  %     test_data(m).trialID
  %         unique trial ID
  %     test_data(m).startHandPos
  %         2x1 vector giving the [x y] position of the hand at the start
  %         of the trial
  %     test_data(m).decodedHandPos
  %         [2xN] vector giving the hand position estimated by your
  %         algorithm during the previous iterations. In this case, N is 
  %         the number of times your function has been called previously on
  %         the same data sequence.
  %     test_data(m).spikes(i,t) (m = trial id, i = neuron id, t = time)
  %     in this case, t goes from 1 to the current time in steps of 20
  %     Example:
  %         Iteration 1 (t = 320):
  %             test_data.trialID = 1;
  %             test_data.startHandPos = [0; 0]
  %             test_data.decodedHandPos = []
  %             test_data.spikes = 98x320 matrix of spiking activity
  %         Iteration 2 (t = 340):
  %             test_data.trialID = 1;
  %             test_data.startHandPos = [0; 0]
  %             test_data.decodedHandPos = [2.3; 1.5]
  %             test_data.spikes = 98x340 matrix of spiking activity

   % Initialize X_test
   numAngles = 8;

   X_test = zeros(1, 98); % Adjust the size based on the number of test trials

    for neuron = 1:98
        X_test(1, neuron) = sum(test_data.spikes(neuron, :));
    end

    % for angle = 1:numAngles
    %     for idx = 1:numTrials % Assuming 20 trials per angle in the test set
    %         row = (angle - 1) * numTrials + idx; % Calculate the row index in X_test
    %         for neuron = 1:98
    %             X_test(row, neuron) = sum(test_data(idx, angle).spikes(neuron, :));
    %         end
    %         labels_test(row) = angle; % Store the corresponding angle in labels
    %     end
    % end

    % Transforming X_test using PCA coefficients from training data
    X_test_centered = X_test - modelParameters.trainMean; % Center the test data, using the mean from the PCA model
    pcaReducedSpikes = X_test_centered * modelParameters.pcaCoeff(:, 1:20); % Apply PCA transformation using the saved coeff
    
    % Applying LDA transformation to get most separable dataset
    ldaTransformedSpikes = ldaTransform(pcaReducedSpikes, modelParameters.ldaModel);
    
    % Predict the angle through our previously trained classification decision tree model
    predictedAngle = predict(modelParameters.treeModel, ldaTransformedSpikes);
    
    % Use the average hand positions for the predicted angle to estimate the hand position
    % The average hand position is stored previously in modelParameters for each angle
    x = modelParameters.avgHandPos(predictedAngle, 1); % x-coordinate
    y = modelParameters.avgHandPos(predictedAngle, 2); % y-coordinate
    return;
end