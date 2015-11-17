function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

% number of training examples
m = size(X, 1);

% You need to return the following variables correctly
p = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters.
%               You should set p to a vector of 0's and 1's
%

h_theta = theta' * X';

for i = 1:m
    %h_theta(i)
    %sigmoid(h_theta(i))
    if (sigmoid(h_theta(i)) >= 0.5)
        p(i) = 1;
    else
        p(i) = 0;
    end
    %p(i)
end

% =========================================================================

end
