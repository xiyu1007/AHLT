

clc;
clear;
close all;
% dataname=["Yale","3Sources","MSRC_v1","NGs","BBCSport","Webkb","Caltech101_7","HW","NUS_WIDE","BDGP","OutdoorScene","MNIST_4"];
% dataname=["Hdigit","Animal", "ALOI", "Reuters", "NUSWIDEOBJ"];
dataname = "Yale";              % Dataset name
method = "AHLT";  

repath = sprintf('output\\%s\\%s\\%s_re.mat', dataname, dataname, method);
if exist(repath,'file')
    result = load(repath).result;
end

result.stats_mean
plotLoss(result.Loss)
plotSpectralEmbedding(result.F, result.pred, result.labs, 3); % 3D embedding



%% ===================== Subfunctions ====================================
function plotLoss(Loss)
	% plotLoss - Plot constraint residuals during AHLT iterations
	%
	% Input:
	%   Loss - 2 x max_iter matrix (or 1 x max_iter)
	numLines = min(2, size(Loss,1));
	Loss = Loss(1:numLines, :);

	figure; hold on;
	for i = 1:numLines
		plot(Loss(i,:), 'LineWidth', 1.5);
	end
	hold off;

	grid on;
	xlabel('Iterations', 'FontSize', 12, 'FontName', 'Times New Roman');
	ylabel('Constraint residuals', 'FontSize', 12, 'FontName', 'Times New Roman');
	set(gca, 'FontSize', 11, 'FontName', 'Times New Roman');
	axis tight;

	% LaTeX legend
	legend({'$\|\mathcal{G} - \mathcal{Z}\|_\infty$', ...
			'$\|\mathcal{H} - \mathcal{A} \times_3 (\mathcal{Z} + \mathcal{D}) - \mathcal{E}\|_\infty$'}, ...
		'Interpreter', 'latex', 'FontSize', 11, 'Location', 'northeast');
end
