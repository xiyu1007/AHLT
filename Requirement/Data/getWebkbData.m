function [X,Y] = getWebkbData()
    wb = load('Datasets\Webkb\webkb.mat');
    Y = categorical(wb.y);
    Y = dummyvar(Y);
    X = wb.X;
    X = NormalizeData(X,1);
end

