function X = Zscore(X,dim)
    if nargin < 2, dim = 1; end
    % X : cell (n*d) 
    if ~iscell(X)
        [X, ~] = Scaler(dim).fit_transform(X);  % 标准化每一列特征
    else
        for m = 1:numel(X)
            X_m = X{m};                 % n x d_m
            [X{m}, ~] = Scaler(dim).fit_transform(X_m);  % 标准化每一列特征
        end
    end
end