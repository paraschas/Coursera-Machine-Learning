function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
% number of features
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly
p = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% size(Theta1) == [25 401]
% size(Theta2) == [10 26]

% assign a_1
a_1 = [ones(m, 1) X];
% size(a_1) == [5000 401]

z_2 = a_1 * Theta1';
% size(z_2) == [5000 25]

a_2 = sigmoid(z_2);
% size(a_2) == [5000 25]

% add ones to a_2
a_2 = [ones(m, 1) a_2];
% size(a_2) == [5000 26]

z_3 = a_2 * Theta2';
% size(z_3) == [5000 10]

a_3 = sigmoid(z_3);
% size(a_3) == [5000 10]

h_theta = a_3;
% size(h_theta) == [5000 10]

[values p] = max(h_theta, [], 2);

% one-liner
%[values p] = max(sigmoid([ones(m, 1) sigmoid([ones(m, 1) X] * Theta1')] * Theta2'), [], 2);

% =========================================================================

end
