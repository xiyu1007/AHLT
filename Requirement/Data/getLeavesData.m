function [X,Y] = getLeavesData()
    leaveas = load('Datasets\Leaves\100Leaves.mat');

    Y = leaveas.Y;
    Y = categorical(Y);
    Y = dummyvar(Y);
    X = leaveas.X;
    X = NormalizeData(X,1);
end



