function [f, df] = ETR(x, delta)
	% ETR - Compute f(x) = exp(delta^2) * x / (delta + x) and its derivative
	%
	% -------------------------------------------------------------------------
	% Input:
	%	x     - input scalar or vector
	%	delta - positive parameter (default: 1)
	%
	% Output:
	%	f  - function value: f(x) = exp(delta^2) * x / (delta + x)
	%	df - derivative:     f'(x) = exp(delta^2) * delta / (delta + x)^2

	x = abs(x);
	if nargin < 2
		delta = 1;  % default value
	end

	% function value
	f = exp(delta^2) .* x ./ (delta + x);

	% derivative
	df = (exp(delta^2) .* delta) ./ (delta + x).^2;
end
