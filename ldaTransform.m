function X_lda = ldaTransform(X_test, model)
    % Center the test data using the overall mean of the training data
    X_centered = X_test - model.meanOverall;
    % Project the centered test data onto the linear discriminants
    X_lda = X_centered * model.linearDiscriminants;
end