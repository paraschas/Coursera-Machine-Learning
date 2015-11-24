function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
% number of training examples
m = length(y);
% m == 12

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));
% size(grad) == [2 1]

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% size(X) == [12 2]
% size(y) == [12 1]
% size(theta) == [2 1]

%h_theta = X * theta;
% size(h_theta) == [12 1]
%J = (1 / (2 * m)) * sum((h_theta - y) .^ 2) + (lambda / (2 * m)) * sum(theta(2:end, :) .^ 2);

J = (1 / (2 * m)) * sum(((X * theta) - y) .^ 2) + (lambda / (2 * m)) * sum(theta(2:end, :) .^ 2);

% =========================================================================

grad = grad(:);

end
