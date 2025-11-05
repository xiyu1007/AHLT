function [X,Y] = getBBCSportData()
    data = load('Datasets\BBCSport\BBCSport.mat');
    Y = data.truelabel{1};
    Y = categorical(Y');
    Y = dummyvar(Y);
    X = {full(data.data{1}'), full(data.data{2}')};
    % X = {data.data{1}', data.data{2}'};
    X = NormalizeData(X,1);
end



