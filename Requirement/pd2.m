function D = pd2(A, B, flag)
% PD2  Compute the pairwise squared Euclidean distance matrix
%
% Overview:
%   This function computes the squared Euclidean distances between all sample
%   vectors in matrices A and B. The output D(i,j) represents the squared 
%   distance between the i-th row of A and the j-th row of B.
%
% -------------------------------------------------------------------------
% Input:
%   A    : n×p matrix
%          - n samples, each row is a p-dimensional feature vector A_i
%   B    : m×p matrix
%          - m samples, each row is a p-dimensional feature vector B_j
%   flag : (optional) indicator
%          - If flag == 1 and n == m, the diagonal elements of D are set to
%            inf (to exclude self-distances)
%
% Output:
%   D    : n×m matrix
%          - D(i,j) = ||A_i - B_j||_2^2
%            squared Euclidean distance between A_i and B_j
% -------------------------------------------------------------------------
% Author : Xi Guo
% Email  : xiguo@my.swjtu.edu.cn
% Date   : 2025-10-10
% -------------------------------------------------------------------------
    if nargin < 2 || isempty(B)
        B = A;
    end
    % Compute squared norms of each row (sample)
    A2 = sum(A.^2, 2);   % n×1 vector
    B2 = sum(B.^2, 2);   % m×1 vector

    % Compute the squared Euclidean distance matrix:
    % (A_i - B_j)^2 = ||A_i||^2 + ||B_j||^2 - 2*A_i·B_j
    D = A2 + B2' - 2 * (A * B');

    [n, ~] = size(A);
    [m, ~] = size(B);

    if n == m
        D(1:n+1:end) = 0; % Set diagonal to inf
    end
    
    % If flag == 1 and sample sizes match, set diagonal elements to inf
    if nargin >= 3 && flag == 1 
        if n == m
            D(1:n+1:end) = 1e6; % Set diagonal to inf
        end
    end
end
