function [X,Y] = getCaltech101_7Data()
    data = load('Datasets\Caltech101-7\Caltech101-7.mat');
    Y = data.y;
    Y = categorical(Y);
    Y = dummyvar(Y);
    X = data.X;
    X = NormalizeData(X,1);
end



