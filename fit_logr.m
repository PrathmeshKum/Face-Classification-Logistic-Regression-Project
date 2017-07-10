% Author: Stefan Stavrev 2013

% Description: MAP logistic regression.
% Input: X - a (D+1)xI data matrix, where D is the data dimensionality
%            and I is the number of training examples. The training
%            examples are in X's columns rather than in the rows, and
%            the first row of X equals ones(1,I).
%        w - a Ix1 vector containing the corresponding world states for
%            each training example,
%        var_prior - scale factor for the prior spherical covariance,
%        X_test - a data matrix containing training examples for which
%                 we need to make predictions,
%        initial_phi - (D+1)x1 vector that represents the start solution.
% Output: predictions - 1xI_test row vector which contains the predicted
%                       class values for the input data in X_test.
%         phi - (D+1)x1 column vector that contains the coefficients for
%               the linear activation function.
function [predictions, phi] = fit_logr (X, w, var_prior, X_test, initial_phi)
    % Find the MAP estimate of the parameters phi.
    options = optimset('GradObj','on','Hessian','on');
    phi = fminunc(@(phi) fit_logr_cost(phi, X, w, var_prior),initial_phi, options);
    
    % Compute the predictions for X_test.
    predictions = sigmoid(phi' * X_test);
end