function prediction = predict_tree(tree, X)
    % Usage:
    % Make predictions using the decision tree
    % Input:
    %   Tree: tree model
    %   X: input matrix
    % Output:
    %   prediction: label prediction
    prediction = zeros(size(X, 1), 1);
    for i = 1:size(X, 1)
        node = tree;
        while ~node.isLeaf
            if X(i, node.splitFeature) <= node.splitValue
                node = node.leftChild;
            else
                node = node.rightChild;
            end
        end
        prediction(i) = node.classLabel;
    end
end