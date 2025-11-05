function [X,Y] = getNGsData()
    data = load('Datasets\NGs\NGs.mat');
    Y = data.truelabel{1};
    Y = categorical(Y');
    Y = dummyvar(Y);
    X = {full(data.data{1}'), full(data.data{2}'), full(data.data{2}')};
    X = NormalizeData(X,1);
    % X = MaxNormalize(X);
    % X = Zscore(X);
end



