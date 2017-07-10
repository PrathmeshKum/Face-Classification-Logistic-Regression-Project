% Author: Stefan Stavrev 2013

% Description: Kernel logistic regression.
% Input: X - a (D+1)xI data matrix, where D is the data dimensionality
%            and I is the number of training examples. The training
%            examples are in X's columns rather than in the rows, and
%            the first row of X equals ones(1,I),
%        w - a Ix1 vector containing the corresponding world states for
%            each training example,
%        var_prior - scale factor for the prior spherical covariance,
%        X_test - a data matrix containing training examples for which
%                 we need to make predictions,
%        initial_psi - Ix1 vector that represents the start solution,
%        kernel - the kernel function,
%        lambda - the parameter used in the Gaussian kernel.
% Output: predictions - 1xI_test row vector which contains the predicted
%                       class values for the input data in X_test.
%         psi - Ix1 column vector that contains the coefficients for
%               the activation function.
function [predictions, psi] = fit_klogr (X, w, var_prior, X_test, initial_psi, kernel, lambda)

    % Compute K[X,X].
    I = size(X,2);
    K = zeros(I,I);
    for i=1:I
        for j=1:I
            K(i,j) = kernel(X(:,i), X(:,j), lambda);
        end
    end
    
    % Compute K[X,X_test].
    I_test = size(X_test,2);
    K_test = zeros(I, I_test);
    for i=1:I
        for j=1:I_test
            K_test(i,j) = kernel(X(:,i), X_test(:,j), lambda);
        end
    end

    % Find the MAP estimate of the parameters psi.
    options = optimset('GradObj','on','Hessian','on');
    psi = fminunc(@(psi) fit_klogr_cost(psi, X, w, var_prior, K),initial_psi, options);
    
    % Compute the Hessian at psi.
    H = diag(repmat(1/var_prior,1,I));
    ys = sigmoid(psi' * K); 
    for i=1:I
        y = ys(i);
        H = H - y * (1-y) * K(:,i) * K(:,i)';
    end
    
    % Set the mean and variance of the Laplace approximation.
    mu = psi;
    invH = H\eye(size(H));
    var = -(invH);
    
    
    % Compute mean and variance of the activation.
    mu_a = mu' * K_test;
    var_a_temp = X * var * K_test;
    var_a = zeros(1,I_test);
    for i=1:I_test
        var_a(i) = X_test(:,i)' * var_a_temp(:,i);
    end
    
    % Approximate the integral to get the Bernoulli parameter.
    predictions = sqrt(1 + pi / 8 .* var_a); 
    predictions = mu_a ./ predictions;
    predictions = sigmoid(predictions);
end