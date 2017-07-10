% Author: Stefan Stavrev 2013

% Description: Cost function for MAP logistic regression.
% Input: phi - (D+1)x1 column vector that contains the coefficients for
%               the linear activation function,
%        X - a (D+1)xI data matrix, where D is the data dimensionality
%            and I is the number of training examples. The training
%            examples are in X's columns rather than in the rows, and
%            the first row of X equals ones(1,I).
%        w - a Ix1 vector containing the corresponding world states for
%            each training example,
%        var_prior - scale factor for the prior spherical covariance.
% Output: L - the value of the cost function evaluated at phi,
%         g - (D+1)x1 gradient vector,
%         H - (D+1)x(D+1) Hessian matrix containing the second derivatives.
function [L, g, H] = fit_logr_cost (phi, X, w, var_prior)    
    I = size(X,2);
    D = size(X,1) - 1;
    
    
    % Initialize.
      %L = I * (-log (mvnpdf (phi, zeros(D+1,1), var_prior*eye(D+1))));
      %L = I * (-logmvnpdf (phi', zeros(D+1,1)', var_prior*eye(D+1)));
      %g = I * phi / var_prior;
      %H = I * diag(repmat(1/var_prior,1,D+1));
      L = 0;
      g = 0;
      H = 0;
    
    predictions = sigmoid(phi' * X); 
    for i = 1 : I
        % Update L.
        y = predictions(i);
     
        if w(i) == 1
            L = L - log(y);
        else
            L = L - log(1-y);
        end
        
        % Update g and H.
        x_i = X(:,i);
        g = g + (y-w(i)) * x_i;
        H = H + y * (1-y) * (x_i * x_i');
    end
    
    disp('Update complete');
    
end