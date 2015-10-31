function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


% compute the hypothesis
hypothesis = sigmoid(X * theta);

% compute the cost function
for i = 1:m
    J = J + (-(y(i)*log(hypothesis(i))) - ((1 - y(i))*log(1 - hypothesis(i))));
end

J = (1/m)*J;

summation = 0;

for j = 2:size(theta)
    summation = summation + theta(j)^2;
end
summation = summation * (lambda/(2*m));

J = J + summation;

% compute the gradient
for i = 1:m
    grad(1) = grad(1) + ((hypothesis(i) - y(i)) * X(i,1));
end
grad(1) = (1/m) * grad(1);

for j = 2:size(theta)
    for i = 1:m
        grad(j) = grad(j) + ((hypothesis(i) - y(i)) * X(i,j));
    end
    grad(j) = (1/m) * grad(j) + (lambda/m)*theta(j);
end

% =============================================================

end
