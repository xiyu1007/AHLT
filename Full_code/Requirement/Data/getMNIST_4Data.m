function [X,Y] = getMNIST_4Data()
    od = load('Datasets\MNIST-4\MNIST-4.mat');
    Y = od.y;
    Y = categorical(Y);
    Y = dummyvar(Y);
    X = od.X;
    X = NormalizeData(X,1);
end



