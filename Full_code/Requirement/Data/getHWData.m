function [X,Y] = getHWData()
    data = load('Datasets\HW\HW.mat');
    Y = data.truelabel{1};
    Y = categorical(Y);
    Y = dummyvar(Y);
    X = cell(1,numel(data.data));
    for m=1:numel(data.data)
        X{m} = data.data{m}';
    end
    X = NormalizeData(X,1);
end



