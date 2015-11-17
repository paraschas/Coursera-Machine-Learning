function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
% number of training examples
m = length(y);
% number of parameters
n = size(theta);

% You need to return the following variables correctly
J = 0;
grad = zeros(n);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h_theta = sigmoid(theta' * X');

% compute the cost
s = 0;
for i = 1:m
    s = s + ((-y(i)) * log(h_theta(i)) - (1 - y(i)) * log(1 - h_theta(i)));
end

s_p = 0;
for i = 2:n
    s_p = s_p + theta(i) ^ 2;
end

J =  (1 / m) * s + (lambda / (2 * m)) * s_p;

% compute the gradient of theta 0
s = 0;
for i = 1:m
    s = s + ((h_theta(i) - y(i)) * X(i, 1));
end
grad(1) =  (1 / m) * s;

% compute the gradient of thetas 1 to n
for j = 2:n
    s = 0;
    for i = 1:m
        s = s + ((h_theta(i) - y(i)) * X(i, j));
    end
    grad(j) =  (1 / m) * s + (lambda / m) * theta(j);
end

%J
%grad

% =============================================================

end
