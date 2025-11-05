function [X,Y] = TransposeXY(X,Y)
    if nargin < 2 || isempty(Y)
        Y = [];
    else
        Y = Y';  % 转置 Y
    end
    for m=1:numel(X)
        X{m} = X{m}';
    end
end

