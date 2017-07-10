function [logp] = logmvnpdf(x,mu,Sigma)
% outputs log likelihood array for observations x  where x_n ~ N(mu,Sigma)
% x is Nxd, mu is 1xd, Sigma is dxd

[N,d] = size(x);
const = -0.5 * d * log(2*pi);
Sigma=nearestSPD(Sigma);

xc = bsxfun(@minus,x,mu);

term1 = -0.5 * sum((xc / Sigma) .* xc, 2); % N x 1
term2 = const - 0.5 * logdet(Sigma);    % scalar
logp = term1' + term2;

end

function y = logdet(A)

U = chol(A);
y = 2*sum(log(diag(U)));

end