% Author: Stefan Stavrev 2013

% Description: Cost function for relevance vector classification.
% Input: psi - Ix1 column vector which is subject to optimization,
%        w - a Ix1 vector containing the corresponding world states for
%            each training example,
%        Hd - Ix1 vector which contains the hidden values,
%        K - the kernel matrix.
% Output: L - the value of the cost function evaluated at psi,
%         g - Ix1 gradient vector,
%         H - IxI Hessian matrix containing the second derivatives.
function [L, g, H] = fit_rvc_cost (psi, w, Hd, K)
    % Initialize.
    I = size(K,1);
    Hd_diag = diag(Hd);
    %L = I * (-log (mvnpdf (psi, zeros(I,1), diag(1 ./ Hd))));    
    L = I * (-logmvnpdf (psi', zeros(I,1)', diag(1 ./ Hd)));
    g = I * Hd_diag * psi;
    H = I * Hd_diag;

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
end