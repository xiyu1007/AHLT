function [X,Y] = getAnimalData()
    data = load('Datasets\Animal\Animal.mat');
    Y = data.gt;
    Y = categorical(Y);
    Y = dummyvar(Y);
    X = {data.X{1}', data.X{2}'};
    X = NormalizeData(X,1);
end



