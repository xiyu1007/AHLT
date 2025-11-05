function [G,f] = TNN(R,ref,dim,fhandle,p,tol,max_iter)
	% TNN - Tensor Nuclear Norm shrinkage in the Fourier domain
	%
	% -------------------------------------------------------------------------
	% Input:
	%   R        - h*n*Views tensor
	%   ref      - scalar or vector, step size / threshold for shrinkage
	%   dim      - dimension along which to perform FFT (0,1,2), default 2
	%   fhandle  - function handle, non-convex shrinkage function (optional)
	%   p        - parameter for fhandle
	%   tol      - tolerance for iterative shrinkage (default 1e-4)
	%   max_iter - max iterations for shrinkage (default 30)
	%
	% Output:
	%   G - tensor after nuclear norm shrinkage
	%   f - objective function value (optional)
	%
	% -------------------------------------------------------------------------
	% Author : Xi Guo
	% Email  : xiguo@my.swjtu.edu.cn
	% Date   : 2025-10-27
	% -------------------------------------------------------------------------

	if nargin < 6, tol = 1e-4; end
	if nargin < 7, max_iter = 30; end
	if nargin < 3 || isempty(dim), dim = 2; end

	% Shift tensor along dim
	R = shiftdim(R, dim);
	[~, ~, n3] = size(R);

	% Expand scalar ref if necessary
	if isscalar(ref)
		ref = repmat(ref, 1, n3);
	end

	% FFT along 3rd dimension
	Rf = fft(R, [], 3); 
	Gf = zeros(size(Rf));
	f = 0;

	% Determine frequency slices for conjugate symmetry
	if mod(n3,2) == 0
		endSlice = n3/2 + 1;
	else
		endSlice = (n3+1)/2;
	end

	for i = 1:endSlice
		[U,S,V] = svd(Rf(:,:,i),'econ');
		s = diag(S);

		% Apply shrinkage (with optional non-convex function)
		if nargin >= 4
			for iter = 1:max_iter
				[~, dfs] = fhandle(s,p);
				s_new = max(s - ref(i)*dfs, 0);
				if norm(s_new - s, inf) < tol
					break;
				end
				s = s_new;
			end
		else
			s_new = max(s - ref(i), 0);
        end

        % 更新目标函数值 
        if nargin >= 4
            f = f + sum(fhandle(s_new,p));
        else
            f = f + sum(s_new);
        end

		% Update current frequency slice
		Gf(:,:,i) = U * diag(s_new) * V';

		% Fill conjugate symmetric slices
		if i > 1
			if mod(n3,2) == 0 && i < n3/2 + 1
				Gf(:,:,n3-i+2) = conj(Gf(:,:,i));
			elseif mod(n3,2) == 1 && i <= (n3-1)/2
				Gf(:,:,n3-i+2) = conj(Gf(:,:,i));
			end
		end
	end

	% Nyquist frequency must be real for even n3
	if mod(n3,2) == 0
		Gf(:,:,n3/2+1) = real(Gf(:,:,n3/2+1));
	end

	% Inverse FFT and shift back
	G = ifft(Gf, [], 3, 'symmetric');
	G = shiftdim(G, 3-dim);
end
