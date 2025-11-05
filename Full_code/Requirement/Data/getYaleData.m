function [X,Y] = getYaleData()
    data = load('Datasets\Yale\Yale.mat');
    Y = data.y;
    Y = categorical(Y);
    Y = dummyvar(Y);
    X = data.X;
    X = NormalizeData(X,1);
end



