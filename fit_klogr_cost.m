% Author: Stefan Stavrev 2013

% Description: Cost function for kernel logistic regression.
% Input: psi - Ix1 column vector that contains the coefficients for
%              the activation function,
%        X - a (D+1)xI data matrix, where D is the data dimensionality
%            and I is the number of training examples. The training
%            examples are in X's columns rather than in the rows, and
%            the first row of X equals ones(1,I).
%        w - a Ix1 vector containing the corresponding world states for
%            each training example,
%        var_prior - scale factor for the prior spherical covariance,
%        K - the kernel matrix.
% Output: L - the value of the cost function evaluated at psi,
%         g - Ix1 gradient vector,
%         H - IxI Hessian matrix containing the second derivatives.
function [L, g, H] = fit_klogr_cost (psi, X, w, var_prior, K)
    % Initialize.
    I = size(X,2);
    %L = I * (-log (mvnpdf (psi, zeros(I,1), var_prior*eye(I))));
    L = I * (-logmvnpdf (psi', zeros(I,1)', var_prior*eye(I)));
    g = I * psi / var_prior;
    H = I * diag(repmat(1/var_prior,1,I));

    predictions = sigmoid(psi' * K);
    for i = 1 : I
        % Update L.
        y = predictions(i);
        if w(i) == 1
            L = L - log(y);
        else
            L = L - log(1-y);
        end
        
        % Update g and H.
        g = g + (y-w(i)) * K(:,i);
        H = H + y * (1-y) * K(:,i) * K(:,i)';
    end
    
    disp('Update complete');
end