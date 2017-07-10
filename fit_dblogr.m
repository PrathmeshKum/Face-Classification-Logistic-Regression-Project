% Author: Stefan Stavrev 2013

% Description: Dual Bayesian logistic regression.
% Input: X - a (D+1)xI data matrix, where D is the data dimensionality
%            and I is the number of training examples. The training
%            examples are in X's columns rather than in the rows, and
%            the first row of X equals ones(1,I),
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
function [predictions, psi] = fit_dblogr (X, w, var_prior, X_test, initial_psi)
    % Find the MAP estimate of the parameters psi.
    options = optimset('GradObj','on','Hessian','on');
    psi = fminunc(@(psi) fit_dlogr_cost1(psi, X, w, var_prior),initial_psi, options);
    
    % Compute the Hessian at psi.
    I = size(X,2);
    H = diag(repmat(1/var_prior,1,I));
    ys = sigmoid((X*psi)' * X); 
    for i=1:I
        temp = X' * X(:,i);
        y = ys(i);
        H = H - y * (1-y) * (temp * temp');
    end
    
    
    % Set the mean and variance of the Laplace approximation.
    mu = psi;
    invH = H\eye(size(H));
    var = -(invH);
    
    % Compute mean and variance of the activation.
    mu_a_temp = X' * X_test;
    mu_a = mu' * mu_a_temp;
    var_a_temp = X * var * mu_a_temp;
    I_test = size(X_test,2);
    var_a = zeros(1,I_test);
    for i=1:I_test
        var_a(i) = X_test(:,i)' * var_a_temp(:,i);
    end
    
    % Approximate the integral to get the Bernoulli parameter.
    lambda = sqrt(1 + pi / 8 .* var_a); 
    lambda = mu_a ./ lambda;
    lambda = sigmoid(lambda);
    
    predictions = lambda;
end