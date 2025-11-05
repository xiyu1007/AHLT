function parameter = paramInit(method)
	% paramInit - Initialize parameters for a given method
	%
	% -------------------------------------------------------------------------
	% Input:
	%	method - string, specifying the method name (e.g., 'AHLT')
	%
	% Output:
	%	parameter - matrix, each row is a combination of parameters
	%
	% -------------------------------------------------------------------------
	% Author : Xi Guo
	% Email  : xiguo@my.swjtu.edu.cn
	% Date   : 2025-10-27
	% -------------------------------------------------------------------------

	switch upper(method)
		case {'AHLT','AHLT_FUN'}
			% Parameter ranges for AHLT
			alphaSpace = [1e-4, 1e-3, 0.01, 0.1, 1];
			betaSpace  = [1e-4, 1e-3, 0.01, 0.1, 1];
			pSpace     = [0.01, 0.1, 1, 10]; % delta parameter for SREP nonconvex proxy
			oSpace     = 2;                  % high-order graph diffusion steps (2 by default, 0 means no diffusion)
			kSpace     = [1, 3, 5, 7, 9];    % number of neighbors for initial graph
			tSpace     = 3;                  % number of anchors (3*c, large datasets: 30)

			paramSpace = {alphaSpace, betaSpace, pSpace, oSpace, kSpace, tSpace};
			parameter  = combvec(paramSpace{:})';
			parameter  = sortrows(parameter, 'descend');

		otherwise
			error('paramInit:UnknownMethod', 'Unknown method: %s', method);
	end
end
