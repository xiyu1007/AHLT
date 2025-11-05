function [X,Y] = getNUS_WIDEData()
    data = load('Datasets\NUS-WIDE\NUS-WIDE.mat');
    Y = data.y;
    Y = categorical(Y);
    Y = dummyvar(Y);
    X = data.X;
    X = NormalizeData(X,1);
end



