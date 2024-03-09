function tree = buildDecisionTree(X, Y, maxDepth)
    % This function initializes the decision tree building process by calling 
    % the growTree function with the provided dataset and parameters.
    % Inputs:
        % X: The feature matrix where each row is a sample and each column is a feature.
        % Y: The target values (labels) corresponding to each sample in X.
        % maxDepth: The maximum depth of the tree, which controls how deep the tree can grow.
    % Output: 
        %  A decision tree (tree) grown from the dataset up to the specified maxDepth.
    tree = growTree(X, Y, maxDepth);
end

function node = growTree(X, Y, maxDepth)
    % Recursively grows a decision tree.
    % If the stopping criteria are met (e.g., reaching maximum depth), it creates a leaf node.
    % Otherwise, it finds the best feature and value to split the data, then creates a new tree node with left and right children being the result of further calls to growTree on the split data.
    % Inputs: 
        % The same as in buildDecisionTree.
    % Output: 
        % A node that is either a leaf or an internal node with links to child nodes.
    if stoppingCriteria(X, Y, maxDepth)
        node = createLeafNode(Y);
    else
        [bestFeature, bestValue] = findBestSplit(X, Y);
        % Partition data based on best split
        [X_left, Y_left, X_right, Y_right] = splitData(X, Y, bestFeature, bestValue);
        % Grow left and right child nodes
        node = TreeNode();
        node.isLeaf = false;
        node.splitFeature = bestFeature;
        node.splitValue = bestValue;
        node.leftChild = growTree(X_left, Y_left, maxDepth - 1);
        node.rightChild = growTree(X_right, Y_right, maxDepth - 1);
    end
end

function isStop = stoppingCriteria(X, Y, maxDepth)
    % Determines whether the tree should stop growing. The given implementation stops growing when the tree reaches the maximum depth.
    % Inputs:
        % X and Y: Current subset of data and labels.
        % maxDepth: Maximum allowable depth of the tree.

    % Output: 
        %  isStop, a boolean indicating whether to stop growing the tree.
    isStop = (maxDepth == 0);
end

function node = createLeafNode(Y)
    % Create a leaf node with majority class label
    % Input: 
        % Y, the labels of the data points in the leaf.
        % Creates a leaf node by determining the majority class label among the data points.
    % Output: 
        % node, a leaf node with the majority class label.
    node = TreeNode();
    node.isLeaf = true;
    node.classLabel = mode(Y);
end

function [bestFeature, bestValue] = findBestSplit(X, Y)
    % Finds the best feature and value to split the data. The implementation 
    % of how to find the best split is not provided in your code and often 
    % involves calculating a measure like information gain or Gini impurity.
    % Inputs:
        % X: Feature matrix.
        % Y: Corresponding labels.
    % Output: 
        % bestFeature, the index of the best feature to split on, and bestValue, the value of that feature to split the data.
    numFeatures = size(X, 2);
    bestFeature = randi(numFeatures);
    bestValue = rand();
end

function [X_left, Y_left, X_right, Y_right] = splitData(X, Y, feature, value)
    % Splits the dataset into two subsets (left and right) based on whether the 
    % data points' feature value is less than or equal to (left) or greater than (right) the given value.
    % Inputs:
        % X, Y: The dataset to split.
        % feature: The feature index to split on.
        % value: The value of the feature to split the dataset.
    % Outputs: 
        % X_left, Y_left, X_right, Y_right - the subsets of the data and labels resulting from the split.
    X_left = X(X(:, feature) <= value, :);
    Y_left = Y(X(:, feature) <= value);
    X_right = X(X(:, feature) > value, :);
    Y_right = Y(X(:, feature) > value);
end