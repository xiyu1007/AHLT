function [S, mu_vec] = KKT_Neighbor_Matrix(D, k, mask)
% KKT_Neighbor_Matrix  Computes sparse neighbor weights based on KKT conditions
%
% -------------------------------------------------------------------------
% Function Overview:
%   This function computes a sparse neighbor representation for each column
%   vector of D by solving the following optimization problem:
%
%       min_s   s^T s + s^T (d/ mu) 
%       s.t.    s^T 1 = 1,  s >= 0
%
%   This is equivalent to:
%
%       min_s   ||s - q||_2^2,   where q = -d/2 / mu
%
%   The goal is to find an optimal mu for each column such that the resulting
%   vector s has exactly k non-zero elements. The output S is column-normalized.
%
% -------------------------------------------------------------------------
% Input:
%   D       : m×n matrix, each column is a feature/sample vector
%   k       : sparsity parameter, number of non-zero elements per column
%
% Output:
%   S       : m×n matrix, each column is the corresponding sparse neighbor
%             weight vector, normalized so that sum(S(:,i)) = 1
%   mu_vec  : 1×n vector of optimal mu values for each column
%
% -------------------------------------------------------------------------
% Author : Xi Guo
% Email  : xiguo@my.swjtu.edu.cn
% Date   : 2025-10-10
% -------------------------------------------------------------------------
    if nargin >= 3
        disMax = max(D,[],'all');
        D(mask) = disMax;
    end

    [m, n] = size(D);
    k = max(2, min([k, m-1])); % Ensure k is within [2, n-1]

    % ---- 1. Sort each column in descending order ----
    [sd, sortid] = sort(D, 1, 'ascend'); % sd contains sorted values of each column

    % ---- 2. Sum of the top k elements ----
    sum_dk = sum(sd(1:k, :), 1); % 1×n vector

    % ---- 3. Compute the feasible range of mu ----
    % Based on KKT conditions:
    % lower_bound = max(0, (1 / 2) * (k * sd(k,:) -  sum_dk ) ); % Ensure non-negative
    upper_bound = (1 / 2) * ( k * sd(k+1,:) - sum_dk ); % ! sd(k+1,:) not sd(k+1)
    
    % mu_vec = mean([lower_bound; upper_bound], 1); % Approximate mu by mean
    mu_vec = upper_bound; % mu

    % ---- 4. Handle special cases ----
    idz = mu_vec <= 1e-10;

    % ---- 5. Compute s vectors ----
    ref = - 1 ./ ( 2 * mu_vec + eps );         % Scale factor to avoid division by zero
    Q = D .* ref;                      % Column-wise scaling
    multiplier = (1/k) - (1/k) * (ref .* sum_dk); % Column-wise shift
    S = max(Q + multiplier, 0);        % Project onto non-negative space

    % ---- 6. Column normalization ----
    S(sortid(1:k,idz),idz) = 1 / k;

    S = S ./ (sum(S, 1) + eps);        % Normalize each column
    
    % noZeroNum = sum(S>0,1);
    % if n > 1000
    %     S = sparse(S);
    % end
end
