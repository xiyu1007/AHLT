function [X,Y] = getHdigitData()
    data = load('Datasets\Hdigit\Hdigit.mat');
    Y = data.truelabel{1};
    Y = Y(:);
    Y = categorical(Y);
    Y = dummyvar(Y);
    X = cell(1,numel(data.data));
    for m=1:numel(data.data)
        X{m} = data.data{m}';
    end
    X = NormalizeData(X,2);
end



