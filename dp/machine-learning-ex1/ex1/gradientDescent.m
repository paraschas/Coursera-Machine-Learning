function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha


% Initialize some useful values

% number of training examples
m = length(y);
% number of parameters
n = length(theta);

J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta.
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %


    summation = zeros(n, 1);

    for i = 1:m
        h_theta = 0;
        for j = 1:n
            h_theta = h_theta + theta(j) * X(i, j);
        end

        for j = 1:n
            summation(j) = summation(j) + (h_theta - y(i)) * X(i, j);
        end
    end

    for j = 1:n
        theta(j) = theta(j) - (alpha * (1 / m)) * summation(j);
    end

    % DEBUG
    %computeCost(X, y, theta)

    % ============================================================

    % Save the cost J in every iteration
    J_history(iter) = computeCost(X, y, theta);
end

% DEBUG
%J_history

end
