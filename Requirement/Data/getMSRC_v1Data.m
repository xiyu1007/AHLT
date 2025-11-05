function [X,Y] = getMSRC_v1Data()
    data = load('Datasets\MSRC_v1\MSRC_v1.mat');
    Y = data.truth;
    Y = categorical(Y);
    Y = dummyvar(Y);
    X = {data.msr1,data.msr2,data.msr3,data.msr4,data.msr5};
    X = NormalizeData(X,1);
end



