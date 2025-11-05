function [X,Y] = get3SourcesData()
    data = load('Datasets\3Sources\3Sources.mat');
    Y = data.y;
    Y = categorical(Y);
    Y = dummyvar(Y);
    X = data.X;
    X = NormalizeData(X,1);
end



