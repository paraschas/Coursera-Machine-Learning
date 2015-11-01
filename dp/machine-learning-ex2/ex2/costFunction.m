function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

%theta' * X'
%sigmoid(theta' * X')
h_theta = sigmoid(theta' * X');

summation = 0;
for i = 1:m
    next_term = -y(i) * log(h_theta(i)) - (1 - y(i)) * log(1 - h_theta(i));
    summation = summation + next_term;
end

J =  (1 / m) * summation;

for j = i:n
    summation = 0;
    for i = 1:m
        summation = summation + (h_theta(i) - y(i)) * X(i, j);
    end
    grad(j) =  (1 / m) * summation;
end

%J
%grad

% =============================================================

end
