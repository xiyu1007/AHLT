function X = NormalizeData(X,dim)
    % X : n * d
    if nargin < 2
        dim = 1; % 默认按行
    end

    if iscell(X)
        for m = 1:numel(X)
            if dim == 1
                X{m} = X{m} ./ max(eps, vecnorm(X{m},2,2));
            else
                X{m} = X{m} ./ max(eps, vecnorm(X{m},2,1));
            end
        end
    else
        if dim == 1
            X = X ./ max(eps, vecnorm(X,2,2));
        else
            X = X ./ max(eps, vecnorm(X,2,1));
        end
    end
end
