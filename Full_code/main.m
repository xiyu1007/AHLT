clc;
close all;
clear;
addpath(genpath('Method_Utils'))
addpath('Method')
addpath(genpath('Requirement'))

% Dataset names
% dataname=["Yale","3Sources","MSRC_v1","NGs","BBCSport","Webkb","Caltech101_7","HW","NUS_WIDE","BDGP","OutdoorScene","MNIST_4"];
% dataname=["Hdigit","Animal", "ALOI", "Reuters", "NUSWIDEOBJ"];
dataname = ["Yale","MSRC_v1","NGs","BBCSport"];
dataname = ["Yale"];
% Methods to evaluate
% methods = ["DSTL", "RCAGL", "ESTMC",  "TLRLF4MVC", "t_SVD_MSC", "EBMGC_GNF", "LSGMC", "FMAGC", "SDTR", "AHLT"];

methods = ["AHLT_FUN"]; % or % methods = ["AHLT"];

repeatNum = 10;
warning('off')

%% ==================== Load Dataset and Normalize ====================
for id = 1:length(dataname)
    [X,Y] = feval(strcat('get',dataname(id),'Data'));

    % Get dataset info
    [n, c, M, d] = getDataInfo(X,Y);
    
    for im = 1:length(methods)
        %% ====================== Parameters Setting =======================
        method = methods(im);
        fprintf("Begin running %s on %s ...\n", method, dataname(id));
        
        rmpath(genpath('Compare'))
        addpath(genpath(['Compare\', char(method)]), '-end')
        params = paramInit(method);
        
        repath = sprintf('output\\%s\\%s\\%s_re.mat', dataname(id), dataname(id), method);
        if exist(repath, 'file')
            result = load(repath).result;
            paramMap = result.paramMap;
        else
            result = struct('acc',0,'nmi',0,'param',0,'repeat',0);
            paramMap = ParamHashSet();
        end
        
        %% ========================= Optimization ==========================
        for ip = 1:size(params,1)
            param = params(ip,:);
            if paramMap.contains(param)
                continue
            end
            tic;
            [pred,F,Loss] = methodRun(method,X,Y,n,c,M,d,param,dataname(id));
            [Me,pLabel] = ClusteringMeasure(pred,Y);
            runtime = toc;
            paramMap.add(param);
            
            if (Me.ACC > result.acc) || (Me.ACC == result.acc && Me.NMI > result.nmi)
                result.acc = Me.ACC;
                result.param = param;
                result.nmi = Me.NMI;
                result.F = F;
                result.pred = pLabel;
                result.labs = onehot2label(Y);
                result.Loss = Loss;
            end

            fprintf("Runtime: %2.2f | Param %s | ACC=%.4f | Best ACC=%.4f\n", ...
                runtime, regexprep(num2str(param), '\s+', '-'), Me.ACC, result.acc);
            
            if mod(ip,100) == 0
                reSave(repath,result,paramMap)
            end
            if result.acc == 1
                break;
            end
        end

        reSave(repath,result,paramMap);
        fprintf("%s-Dataset: %s, Best ACC: %.4f, Param: [%s]\n\n", method, dataname(id), result.acc, regexprep(num2str(result.param), '\s+', '-'));
        
        %% ================= Repeat Best Parameter repeatNum Times =========
        if result.repeat == 0
            fprintf("\nRunning best parameter [%s] for %d repetitions...\n", regexprep(num2str(result.param), '\s+', '-'), repeatNum);
            
            % Initialize metrics list
            metrics_list(repeatNum) = struct('ACC',[],'NMI',[],'Purity',[],'PRE',[], ...
                'REC',[],'F',[],'ARI',[],'labs',[],'pred',[]);
            stats_mean = struct();
            stats_std  = struct();

            tic;
            for rep = 1:repeatNum
                [pred,F,Loss] = methodRun(method,X,Y,n,c,M,d,result.param,dataname(id));
                [Me,pLabel] = ClusteringMeasure(pred,Y);
                metrics_list(rep) = Me;
                fprintf("Repetition %2d: ACC=%.4f | NMI=%.4f | Purity=%.4f | F=%.4f | ARI=%.4f\n", ...
                    rep, Me.ACC, Me.NMI, Me.Purity, Me.F, Me.ARI);
            end
            runtime = toc / repeatNum;

            % Compute mean and std of metrics
            fields = fieldnames(metrics_list);
            for f = 1:length(fields)
                fname = fields{f};
                if isnumeric([metrics_list.(fname)]) && isscalar(metrics_list(1).(fname))
                    values = [metrics_list.(fname)];
                    stats_mean.(fname) = mean(values);
                    stats_std.(fname)  = std(values);
                end
            end
            result.acc = stats_mean.ACC;
            result.nmi = stats_mean.NMI;
            result.stats_mean = stats_mean;
            result.stats_std = stats_std;
            result.runtime = runtime;
            result.repeat = 1;

            fprintf("\nDataset: %s | Best Param: [%s]\n", dataname(id), num2str(result.param));
            fprintf("%s Final Results Mean(Std):\n", method);
            fprintf("ACC=%.4f(%.4f) | NMI=%.4f(%.4f) | Purity=%.4f(%.4f) | F=%.4f(%.4f) | ARI=%.4f(%.4f)\n\n\n", ...
                stats_mean.ACC, stats_std.ACC, stats_mean.NMI, stats_std.NMI, ...
                stats_mean.Purity, stats_std.Purity, stats_mean.F, stats_std.F, ...
                stats_mean.ARI, stats_std.ARI);
        end

        % Save final result
        reSave(repath, result, paramMap);

        % Plot first embedding result
        % plotSpectralEmbedding(result.F, result.pred, result.labs, 3);

        % Remove method path
        rmpath(genpath(['Compare\', char(method)]))
    end
end

%% ==================== Subfunction: Run Method ====================
function [pred,F,Loss] = methodRun(method,X,Y,n,c,M,d,param,dataname)
	Loss = []; F = [];
	switch upper(method)
		case 'AHLT'
			[~,pred,F,Loss] = feval(method).run(X,Y,param);
		case 'AHLT_FUN'
			[pred,F,Loss] = feval(method,X,Y,param);
		case 'ESTMC'
			X = TransposeXY(X);
			[pred,F] = feval(method,X,c,param(1),param(2),c*param(3),4);
		case 'DSTL'
			X = TransposeXY(X);
			[pred,F] = feval(method,X,Y,param);
		case 'SDTR'
			[pred,F] = feval(method,X,Y,param);
		case 'FMAGC'
			[pred,F] = feval(method,X,Y,param);
		case 'T_SVD_MSC'
			X = TransposeXY(X);
			[pred,F] = feval(method,X,Y,param);
		case 'EBMGC_GNF'
			[pred,F] = feval(method,X,Y,param);
		case 'TLRLF4MVC'
			[pred,F] = feval(method,X,Y,param,dataname);
		case 'LSGMC'
			X = TransposeXY(X);
			[pred,F] = feval(method,X,Y,param);
		case 'RCAGL'
			[pred,F] = feval(method,X,Y,param);
		otherwise
			error('Unknown method: %s', method);
	end
end

%% ==================== Subfunction: Save Result ====================
function reSave(repath,result,paramMap)
	result.paramMap = paramMap;
	[parentDir,~,~] = fileparts(repath);
	if ~isfolder(parentDir)
		mkdir(fullfile(pwd,parentDir));
	end
	save(repath,'result');
end

