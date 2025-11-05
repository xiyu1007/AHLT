function [X,Y] = getOutdoorSceneData()
    od = load('Datasets\OutdoorScene\OutdoorScene.mat');
    Y = od.y;
    Y = categorical(Y);
    Y = dummyvar(Y);
    X = od.X;
    X = NormalizeData(X,1);
end



