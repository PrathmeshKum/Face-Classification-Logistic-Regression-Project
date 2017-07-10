% Author: Stefan Stavrev 2013

% Description: MAP dual logistic regression.
% Input: X - a (D+1)xI data matrix, where D is the data dimensionality
%            and I is the number of training examples. The training
%            examples are in X's columns rather than in the rows, and
%            the first row of X equals ones(1,I).
%        w - a Ix1 vector containing the corresponding world states for
%            each training example,
%        var_prior - scale factor for the prior spherical covariance,
%        X_test - a data matrix containing training examples for which
%                 we need to make predictions,
%        initial_psi - Ix1 vector that represents the start solution.
% Output: predictions - 1xI_test row vector which contains the predicted
%                       class values for the input data in X_test.
%         psi - Ix1 column vector that contains the coefficients for
%               the activation function.
function [predictions, psi] = fit_dlogr (X, w, var_prior, X_test, initial_psi)
    % Find the MAP estimate of the parameters psi.
    options = optimset('GradObj','on','Hessian','on');
    psi = fminunc(@(psi) fit_dlogr_cost(psi, X, w, var_prior),initial_psi, options);
    
    % Compute the predictions for X_test.
    predictions = sigmoid((X*psi)' * X_test);
end