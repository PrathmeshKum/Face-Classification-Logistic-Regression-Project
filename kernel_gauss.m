% Gaussian kernel function.
% Input: x_i - a column vector,
%        x_j - a column vector.
function f = kernel_gauss (x_i, x_j, lambda)
    x_diff = x_i - x_j;    
    temp = x_diff' * x_diff;    
    f = exp(-0.5*temp/lambda^2);
end