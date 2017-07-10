% Author: Stefan Stavrev 2013

function f = sigmoid (x)
    f = 1 ./ (1 + exp(-x));
end