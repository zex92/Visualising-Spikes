classdef TreeNode
    %Leaf:
        % Type: Boolean
        % Description: Indicates whether the node is a leaf node (true) or an internal node (false).
    % splitFeature:
        % Type: Integer/Variable
        % Description: Represents the feature index on which the node splits the data. Relevant only for internal nodes.
    % splitValue:
        % Type: Numeric
        % Description: The value of the splitFeature used to split the data. Data points with feature value less than or equal to splitValue go to the left child, and others to the right child. Relevant only for internal nodes.
    % leftChild:
        % Type: TreeNode object
        % Description: The left child node of the current node. It represents the subset of data where the splitFeature value is less than or equal to splitValue.
    % rightChild:
        % Type: TreeNode object
        % Description: The right child node of the current node. It represents the subset of data where the splitFeature value is greater than splitValue.
    % classLabel:
        % Type: Variable (typically integer or string)
        % Description: The class label assigned to this node. It is used only for leaf nodes to make a prediction.
    properties
        isLeaf
        splitFeature
        splitValue
        leftChild
        rightChild
        classLabel
    end
end
