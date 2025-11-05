function S = Diffusion(S, k, w)
	% Diffusion - Multi-view similarity diffusion
	%
	% -------------------------------------------------------------------------
	% Overview:
	%	Perform diffusion across multiple similarity matrices (views). Each
	%	iteration computes a weighted global matrix and updates all views.
	%
	% -------------------------------------------------------------------------
	% Input:
	%	S : 1×M cell array, each S{m} is an N×N similarity matrix
	%	k : number of diffusion iterations
	%	w : 1×M weight vector (optional, default uniform)
	%
	% Output:
	%	S : 1×M cell array of diffused similarity matrices
	%
	% -------------------------------------------------------------------------
	% Author : Xi Guo
	% Email  : xiguo@my.swjtu.edu.cn
	% Date   : 2025-10-27
	% -------------------------------------------------------------------------

	M = numel(S);
	[~, n] = size(S{1});

	% Default weights
	if nargin < 3 || isempty(w)
		w = ones(1, M) / M;
	else
		w = w / sum(w);
	end

	% Iterative diffusion
	for t = 1:k
		% Global weighted similarity
		P = zeros(n, n);
		for m = 1:M
			P = P + w(m) * S{m};
		end
		% Update each view
		for m = 1:M
			S{m} = S{m} * P;
		end
	end
end
