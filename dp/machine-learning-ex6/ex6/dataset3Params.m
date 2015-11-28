function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example,
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using
%        mean(double(predictions ~= yval))
%

% DEBUG
%test_values = [0.1 1];
test_values = [0.01 0.03 0.1 0.3 1 3 10 30];

% measure the time taken by the for loop
% start
% http://www.mathworks.com/help/matlab/ref/tic.html
%tic

errors = [];
for i = 1:length(test_values)
    C = test_values(i);
    for j = 1:length(test_values)
        sigma = test_values(j);
        % DEBUG
        %fprintf('training the SVM with C = %f and sigma = %f\n', C, sigma);
        % function [model] = svmTrain(X, Y, C, kernelFunction, tol, max_passes)
        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        % function pred = svmPredict(model, X)
        predictions = svmPredict(model, Xval);
        % http://stackoverflow.com/questions/781410/appending-a-vector-to-an-empty-matlab-matrix
        errors(end+1, :) = [C sigma mean(double(predictions ~= yval))];
    end
end

% measure the time taken by the for loop
% end
% http://www.mathworks.com/help/matlab/ref/toc.html
%toc

[min_error index] = min(errors(:, 3));
% min_error == 0.03
% index == 35
C = errors(index, 1);
% C == 1.0
sigma = errors(index, 2);
% sigma == 0.1

% DEBUG
%errors
%fprintf('minimum error is %f at index %f with C = %f and sigma = %f\n', min_error, index, C, sigma);

% =========================================================================

end
