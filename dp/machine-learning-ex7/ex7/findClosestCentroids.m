function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

%X
% size(X) == [300 2]

% number of examples
m = size(X, 1);
% m == 300

%centroids
% size(centroids) == [3 2]

% number of centroids
K = size(centroids, 1);
% K == 3

% You need to return the following variables correctly.
idx = zeros(size(X, 1), 1);
% size(idx) == [300 1]

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

for i = 1:m
    temp = [];
    for j = 1:K
        temp = [temp; X(i, :)];
    end
    [minimum, idx(i)] = min(sum((temp - centroids) .^ 2, 2));
end

% =============================================================

end
