% Author: Stefan Stavrev 2013

% Description: Relevance vector classification.
% Input: X - a (D+1)xI data matrix, where D is the data dimensionality
%            and I is the number of training examples. The training
%            examples are in X's columns rather than in the rows, and
%            the first row of X equals ones(1,I),
%        w - a Ix1 vector containing the corresponding world states for
%            each training example,
%        nu - degrees of freedom,
%        X_test - a data matrix containing training examples for which
%                 we need to make predictions,
%        initial_psi - Ix1 vector that represents the start solution,
%        kernel - the kernel function,
%        lambda - the parameter used in the Gaussian kernel.
% Output: predictions - 1xI_test row vector which contains the predicted
%                       class values for the input data in X_test,
%         relevant_points - Ix1 boolean vector where a 1 at position i
%                           indicates that point X(:,i) remained after
%                           the elimination phase, that is, it is relevant.
function [predictions, relevant_points] = fit_rvc (X, w, nu, X_test, initial_psi, kernel, lambda)
    % Compute K[X,X].
    I = size(X,2);
    K = zeros(I,I);
    for i=1:I
        for j=1:I
            K(i,j) = kernel(X(:,i), X(:,j), lambda);
        end
    end
    
    % Initialize H.
    H = ones(I,1);
    
    % The main loop.
    iterations_count = 0;
    mu = 0;
    sig = 0;
    options = optimset('GradObj','on','Hessian','on');
    while true     
        % Find the MAP estimate of the parameters psi.
        psi = fminunc(@(psi) fit_rvc_cost(psi, w, H, K),initial_psi, options);
      
        % Compute Hessian S at peak.
        S = diag(H);
        ys = sigmoid(psi' * K);
        for i = 1 : I
            y = ys(i);
            S = S + y * (1-y) * K(:,i) * K(:,i)';
        end
        
        % Set mean and variance of Laplace approximation.
        mu = psi;
        invS = S\eye(size(S));
        sig = -(invS);
        
        % Update H.
        H = H .* diag(sig);
        H = nu + 1 - H;
        H = H ./ (mu.^2 + nu);

        iterations_count = iterations_count + 1;
        disp(['iteration ' num2str(iterations_count)]);
        if (iterations_count == 20)
            break;
        end
    end
    
    % Prune step. Remove row i of mu, col i of X, row and col i of sig, where
    % H(i) > threshold.
    threshold = vpa(2.6150,4);
    selector = vpa(H,4) < threshold;
    X = X(:,selector);
    mu = mu(selector);
    sig = sig(selector, selector);
    relevant_points = selector;
    disp(H);
    
    % Recompute K[X,X].
    I = size(X,2);
    K = zeros(I,I);
    for i=1:I
        for j=1:I
            K(i,j) = kernel(X(:,i), X(:,j), lambda);
        end
    end
    
    % Recompute K[X,X_test].
    I_test = size(X_test,2);
    K_test = zeros(I, I_test);
    for i=1:I
        for j=1:I_test
            K_test(i,j) = kernel(X(:,i), X_test(:,j), lambda);
        end
    end
    
    % Compute mean and variance of activation.
    mu_a = mu' * K_test;
    var_a_temp = sig * K_test;
    var_a = zeros(1,I_test);
    for i=1:I_test
        var_a(i) = K_test(:,i)' * var_a_temp(:,i);
    end
    
    % Approximate the integral to get the Bernoulli parameter.
    predictions = sqrt(1 + pi / 8 .* var_a); 
    predictions = mu_a ./ predictions;
    predictions = sigmoid(predictions);    
end