function [f, df] = SREP(x, delta)
	% SREP - Compute f(x) = 1 - exp(-sqrt(x/delta)) and its derivative
	%
	% -------------------------------------------------------------------------
	% Input:
	%	x     - input scalar or vector
	%	delta - positive parameter (default: 1)
	%
	% Output:
	%	f  - function value: f(x) = 1 - exp(-sqrt(x/delta))
	%	df - derivative:     f'(x) = exp(-sqrt(x/delta)) / (2*sqrt(delta*x))
	%
	% -------------------------------------------------------------------------
	% Author : Xi Guo
	% Email  : xiguo@my.swjtu.edu.cn
	% Date   : 2025-10-27
	% -------------------------------------------------------------------------

	if nargin < 2
		delta = 1;  % default value
	end
	x = x(:);
	x = abs(x);  % avoid negative input

	% function value
	f = 1 - exp(-sqrt(x / delta));

	% derivative
	df = zeros(size(x));
	nonzero_idx = x > 0;
	df(nonzero_idx) = exp(-sqrt(x(nonzero_idx) / delta)) ./ (2 * sqrt(delta * x(nonzero_idx)));

	% handle x = 0 case (optional)
	df(x == 0) = inf;
end
