function T = tensor(dim1, dim2, dim3, type, varargin)
	% TENSOR - Create a 3D tensor with various initialization options
	%
	% -------------------------------------------------------------------------
	% Input:
	%	dim1, dim2, dim3 - tensor dimensions (positive integers)
	%	type             - initialization type: 
	%	                   'zeros','ones','rand','randn','const','orth','eye'
	%	Optional name-value pairs:
	%	   'seed', value    - set random seed
	%
	% Output:
	%	T - 3D tensor of size dim1×dim2×dim3
	%
	% -------------------------------------------------------------------------
	% Author : Xi Guo
	% Email  : xiguo@my.swjtu.edu.cn
	% Date   : 2025-10-27
	% -------------------------------------------------------------------------

	% Check dimensions
	dims = [dim1, dim2, dim3];
	if any(dims <= 0) || any(fix(dims) ~= dims)
		error('Dimensions dim1, dim2, dim3 must be positive integers.');
	end

	% Default type
	if nargin < 4 || isempty(type)
		type = 'zeros';
	end

	% Process random seed
	idxSeed = find(strcmpi(varargin, 'seed'), 1);
	if ~isempty(idxSeed) && idxSeed < numel(varargin)
		rng(varargin{idxSeed + 1});
		varargin([idxSeed, idxSeed + 1]) = [];
	end

	% Tensor initialization
	switch lower(type)
		case 'zeros'
			T = zeros(dim1, dim2, dim3);
		case 'ones'
			T = ones(dim1, dim2, dim3);
		case 'const'
			if isempty(varargin)
				error('''const'' type requires a scalar value as fifth argument.');
			end
			val = varargin{1};
			T = val * ones(dim1, dim2, dim3);
		case 'rand'
			T = rand(dim1, dim2, dim3);
		case 'randn'
			T = randn(dim1, dim2, dim3);
		case 'orth'
			T = zeros(dim1, dim2, dim3);
			for k = 1:dim3
				A = randn(dim1, dim2);
				if dim1 < dim2
					[Q, ~] = qr(A', 'econ');
					Q = Q';
				else
					[Q, ~] = qr(A, 'econ');
				end
				T(:, :, k) = Q;
			end
		case 'eye'
			T = zeros(dim1, dim2, dim3);
			I = eye(dim1, dim2);
			for k = 1:dim3
				T(:, :, k) = I;
			end
		otherwise
			error('Unsupported type: %s', type);
	end
end
