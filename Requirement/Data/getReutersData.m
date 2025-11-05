function [X,Y] = getReutersData()
    data = load('Datasets\Reuters\Reuters.mat');
    Y = data.Y;
    Y = categorical(Y);
    Y = dummyvar(Y);
    X = cell(1,numel(data.X));
    for m=1:numel(data.X)
        X{m} = data.X{m};
    end
    X = NormalizeData(X,2);
end



