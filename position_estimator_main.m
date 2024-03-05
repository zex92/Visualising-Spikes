%% Load the data
load('monkeydata_training.mat')

% Set random number generator for reproducing the results
rng(2013);
ix = randperm(length(trial));

trainingData = trial(ix(1:50),:); % Data is 50x8 struct
testData = trial(ix(51:end),:); % Data is 50x8 struct

% The spike train recorded from the ith unit on the nth trial of the kth
% reaching angle is accessed as: trial(n,k).spikes(i,:) where i = 1...98,
% n = 1...100 and k = 1...8

% Spikes for the first trial for 1:8 reaching angles. It does not matter
% whether we choose 1:8 or 1. The results is the same.
spikes_trial_1 = trainingData(1,1).spikes; % size is 98 neural units x 632 time steps
% Hand position for the first trial for 1:8 reaching angles
hand_pos_trial_1 = trainingData(1,1).handPos; % size is 3 coordinates x 632 times steps
hand_pos_trial_1_clean = hand_pos_trial_1(1:2,:); % size is 2 coordinates (X,Y) x 632 times steps

%% Preparing the data for the model

% Problem:
% We have are 98 features or neural units activated over 632 time steps. 
% Using the information from the 100 of these trials.
% We want to predict continuously the X,Y coordinates.

% The model below expects a 3D array. With dimensions being 
% [sequence length, num_features, batch_size]
% - sequence_length is the number of time steps per trial = 632.
% - num features is the number of neural units per trial = 98.
% - batch_size is the number of sequences to process at a time. In this case
%   the number of trials.

% Parameters
numTrials = 50;
numAngles = 8;
numFeatures = 98;

% Initialize the 3D array for spikes
spikesInput = zeros(maxSequenceLength, numFeatures, numTrials);

% Process each trial
for trialIdx = 1:numTrials
    aggregatedSpikes = zeros(maxSequenceLength, numFeatures);

    for angleIdx = 1:numAngles
        currentSpikes = trainingData(trialIdx, angleIdx).spikes';
        currentLength = size(currentSpikes, 1);
        
        % Pad if necessary
        if currentLength < maxSequenceLength
            currentSpikes = [currentSpikes; zeros(maxSequenceLength - currentLength, numFeatures)];
        end

        % Aggregate spikes data
        aggregatedSpikes = aggregatedSpikes + currentSpikes;
    end

    % Average the spikes across angles
    aggregatedSpikes = aggregatedSpikes / numAngles;

    % Assign to spikesInput
    spikesInput(:, :, trialIdx) = aggregatedSpikes;
end


%% Model

numFeatures = size(spikes_trial_1,1);
numOutputs = 2; % Predicting X and Y coordinates
numFilters = 64;
filterSize = 5;
dropoutFactor = 0.005;
numBlocks = 4;

layer = sequenceInputLayer(numFeatures, Normalization="rescale-symmetric", Name="input");
lgraph = layerGraph(layer);

outputName = layer.Name;

for i = 1:numBlocks
    dilationFactor = 2^(i-1);
    
    layers = [
        convolution1dLayer(filterSize, numFilters, DilationFactor=dilationFactor, Padding="causal", Name="conv1_"+i)
        layerNormalizationLayer
        spatialDropoutLayer(dropoutFactor)
        convolution1dLayer(filterSize, numFilters, DilationFactor=dilationFactor, Padding="causal")
        layerNormalizationLayer
        reluLayer
        spatialDropoutLayer(dropoutFactor)
        additionLayer(2, Name="add_"+i)];

    % Add and connect layers
    lgraph = addLayers(lgraph, layers);
    lgraph = connectLayers(lgraph, outputName, "conv1_"+i);

    % Skip connection
    if i == 1
        layer = convolution1dLayer(1, numFilters, Name="convSkip");
        lgraph = addLayers(lgraph, layer);
        lgraph = connectLayers(lgraph, outputName, "convSkip");
        lgraph = connectLayers(lgraph, "convSkip", "add_" + i + "/in2");
    else
        lgraph = connectLayers(lgraph, outputName, "add_" + i + "/in2");
    end
    
    % Update layer output name
    outputName = "add_" + i;
end

layers = [
    fullyConnectedLayer(numOutputs, Name="fc")
    regressionLayer];

lgraph = addLayers(lgraph, layers);
lgraph = connectLayers(lgraph, outputName, "fc");


%% Train model

options = trainingOptions("adam", ...
    MaxEpochs=60, ...
    miniBatchSize=1, ...
    Plots="training-progress", ...
    Verbose=0);

net = trainNetwork(XTrain,TTrain,lgraph,options);