function [model] = ldaFromScratch(X, labels)
    % Usage:
    % X: Data matrix, where each row is a sample, and each column is a feature
    % labels: Corresponding class labels for each sample

    class_labels = unique(labels);
    n_classes = length(class_labels);
    n_features = size(X, 2);

    % Computing vector means for each label, this essentially finds the
    % feature space that best separates the classes
    mean_vectors = zeros(n_classes, n_features);
    for i = 1:n_classes
        mean_vectors(i, :) = mean(X(labels == class_labels(i), :));
    end

    % Computing:
    % Within-class scatter matrix (how much each class varies within
    % itself)
    S_W = zeros(n_features, n_features);
    for i = 1:n_classes
        class_scatter = zeros(n_features, n_features);
        samples = X(labels == class_labels(i), :);
        for j = 1:size(samples, 1)
            class_scatter = class_scatter + (samples(j, :) - mean_vectors(i, :))' * (samples(j, :) - mean_vectors(i, :));
        end
        S_W = S_W + class_scatter;
    end

    % Overall mean
    mean_overall = mean(X, 1);

    % Computing:
    % Between-class scatter matrix (how much the classes are apart from
    % each other)
    S_B = zeros(n_features, n_features);
    for i = 1:n_classes
        n = sum(labels == class_labels(i)); % Number of samples in class i
        mean_diff = (mean_vectors(i, :) - mean_overall)';
        S_B = S_B + n * (mean_diff * mean_diff');
    end

    % Solving the eigenvalues and eigenvectors, this finds the directions
    % of the feature space that best separates the classes.
    [eigenvectors, eigenvalues] = eig(S_B, S_W);

    % Select the most important linear discriminants eigenvectors
    % Sorting them by decreasing eigenvalues
    [sorted_eigenvalues, sort_order] = sort(diag(eigenvalues), 'descend');
    eigenvectors = eigenvectors(:, sort_order);

    k = min(n_classes - 1, n_features);
    linear_discriminants = eigenvectors(:, 1:k);

    % Saving the model parameters
    model.meanVectors = mean_vectors;
    model.linearDiscriminants = linear_discriminants;
    model.meanOverall = mean_overall;

end

