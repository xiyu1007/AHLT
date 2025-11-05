function X = MaxNormalize(X)
    % X : cell (n*d) 或 numeric 矩阵
    % 将每一列特征缩放到 [0, 1]
    
    if ~iscell(X)
        % 对普通矩阵处理
        X_max = max(X, [], 1);
        X = X ./ (X_max + eps);  % 避免除零
    else
        % 对 cell 数组每个元素处理
        for m = 1:numel(X)
            X_m = X{m};
            X_max = max(X_m, [], 1);
            X{m} = X_m ./ (X_max + eps);  % 避免除零
        end
    end
end
