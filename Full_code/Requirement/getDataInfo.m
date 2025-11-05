function [n, c, M, d] = getDataInfo(X, Y)
    M = numel(X);
    if isvector(Y)
        n = numel(Y);
        c = numel(unique(Y));
    else
        [n, c] = size(Y);
    end
    d = zeros(1, M); % 预分配d的大小
    for m = 1:M
        d(m) = size(X{m}, 2);
    end
end