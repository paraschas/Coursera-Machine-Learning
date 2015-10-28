function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
derivative_1 = 0;
derivative_2 = 0;

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta.
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    % find derivatives for theta vector
    for i = 1:m
        h = theta(1)*X(i,1) + theta(2)*X(i,2);
        % derivative for j(theta1) and j(theta2)
        derivative_1 = derivative_1 + (h - y(i))*X(i,1);
        derivative_2 = derivative_2 + (h - y(i))*X(i,2);
    end

    % compute new thetas
    theta(1) = theta(1) - (alpha*(1/m))*derivative_1;
    theta(2) = theta(2) - (alpha*(1/m))*derivative_2;

    % initialize derivatives
    derivative_1 = 0;
    derivative_2 = 0;

    % ============================================================

    % Save the cost J in every iteration
    J_history(iter) = computeCost(X, y, theta);


end

end
