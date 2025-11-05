function [pred, U_norm] = SpectralClustering(S, k, isEmbedding, iter)
    % SpectralClustering  Spectral clustering based on graph Laplacian
    %
    % -------------------------------------------------------------------------
    % Function Overview:
    %   This function performs spectral clustering using the symmetric normalized
    %   Laplacian. It supports either an input similarity matrix or a precomputed
    %   embedding (feature matrix). By normalizing the first k eigenvectors row-wise
    %   and applying k-means, it assigns cluster labels to samples.
    %
    % -------------------------------------------------------------------------
    % Input:
    %   S           : n×n similarity matrix or n×d embedding matrix
    %   k           : number of clusters (optional, default = 2)
    %   isEmbedding : flag indicating if S is already an embedding (optional, default = 0)
    %   iter        : maximum iterations for k-means (optional, default = 100)
    %
    % Output:
    %   pred      : n×1 vector of cluster labels
    %   U_norm    : n×k matrix of row-normalized eigenvectors or embeddings
    %
    % -------------------------------------------------------------------------
    % Algorithm Details:
    %   If isEmbedding = 0:
    %     1) Symmetrize the similarity matrix S.
    %     2) Construct the symmetric normalized Laplacian:
    %        L_sym = I - D^(-1/2) * S * D^(-1/2)
    %     3) Compute the k eigenvectors corresponding to the smallest eigenvalues.
    %     4) Row-normalize the eigenvectors.
    %     5) Apply k-means to the row-normalized vectors to obtain cluster labels.
    %
    %   If isEmbedding = 1:
    %     - Directly use the input as embedding features, row-normalize, and
    %       apply k-means clustering.
    %
    % -------------------------------------------------------------------------
    % Author : Xi Guo
    % Email  : xiguo@my.swjtu.edu.cn
    % Date   : 2025-10-10
    % -------------------------------------------------------------------------

    if nargin < 2, k = 2; end
    if nargin < 3, isEmbedding = 0; end
    if nargin < 4, iter = 100; end
    % rng('default')
    % rng(42);  % Set random seed for reproducibility

    if ~isEmbedding
        % Symmetrize the similarity matrix
        S = (S + S') / 2;

        % Construct symmetric normalized Laplacian: L_sym = I - D^(-1/2) * S * D^(-1/2)
        D_inv_sqrt = diag(1 ./ sqrt(sum(S,2) + eps));
        L = eye(size(S,1)) - D_inv_sqrt * S * D_inv_sqrt;

        % Compute the k eigenvectors corresponding to the smallest eigenvalues
        [U, ~] = eigs(L, k, 'SA'); % 'SA' = smallest algebraic eigenvalues
    else
        U = S; % Use the input as embedding directly
    end

    % Row-normalize U
    U_norm = bsxfun(@rdivide, U, sqrt(sum(U.^2, 2)));
    U_norm = real(U_norm);
    
    % Apply k-means clustering on the row-normalized vectors
    pred = kmeans(U_norm, k, 'Start', 'plus', 'Replicates', 10, ...
                  'MaxIter', iter, 'EmptyAction', 'singleton');
end
