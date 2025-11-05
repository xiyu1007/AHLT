function F_plot = plotSpectralEmbedding(F, pred, labs, dim)
	% plotSpectralEmbedding - Visualize spectral embedding
	%
	% -------------------------------------------------------------------------
	% Input:
	%	F    - n*d matrix of spectral embeddings
	%	pred - n*1 predicted labels
	%	labs - n*1 true labels (optional)
	%	dim  - dimension for visualization (2 or 3)
	%
	% Output:
	%	F_plot - reduced embedding used for plotting
	%
	% -------------------------------------------------------------------------
	% Author : Xi Guo
	% Email  : xiguo@my.swjtu.edu.cn
	% Date   : 2025-10-27
	% -------------------------------------------------------------------------

	if nargin < 2
		error('Provide at least F and pred.');
	end
	if nargin < 4
		dim = 2; % default 2D
    end
    
    dim = min(3,dim);

	[~, d] = size(F);

	% Dimensionality reduction if necessary
	if d > dim
		F_plot = tsne(F, 'NumDimensions', dim, 'Perplexity', 30);
    else
        dim = min(d,dim);
		F_plot = F(:,1:dim);
	end
    F_plot = (F_plot - min(F_plot, [], 1)) ./ (max(F_plot, [], 1) - min(F_plot, [], 1));
    
	unique_labels = unique(pred);
	colors = lines(length(unique_labels));

	pointSize = 10;

	figure; hold on;


	if nargin >= 3 && ~isempty(labs)
		for i = 1:length(unique_labels)
			label = unique_labels(i);
			idx = (labs == label) & (labs == pred);
			if any(idx)
                if dim == 3
					scatter3(F_plot(idx,1), F_plot(idx,2), F_plot(idx,3), pointSize, colors(i,:), 'filled');
                    view(45,25);
                else
        			scatter(F_plot(idx,1), F_plot(idx,2), pointSize, 'filled', 'MarkerFaceColor', colors(i,:));
                end
                hold on;
			end
        end
		wrong_idx = labs ~= pred;
		for i = 1:length(unique_labels)
			label = unique_labels(i);
			idx = wrong_idx & (labs == label);
			if any(idx)
                if dim == 3
					scatter3(F_plot(idx,1), F_plot(idx,2), F_plot(idx,3), 3*pointSize, colors(i,:), 'x', 'LineWidth', 2);
                else
        			scatter(F_plot(idx,1), F_plot(idx,2), 3*pointSize, 'x', 'MarkerEdgeColor', colors(i,:), 'LineWidth', 2);
                end
                hold on;
			end
		end
	else
		for i = 1:length(unique_labels)
			idx = pred == unique_labels(i);
            if dim == 3
    			scatter3(F_plot(idx,1), F_plot(idx,2), F_plot(idx,3), pointSize, colors(i,:), 'filled');
            else
                scatter(F_plot(idx,1), F_plot(idx,2), pointSize, 'filled', 'MarkerFaceColor', colors(i,:));
            end
		end
	end

    axis equal;
	xlabel('Dimension 1');
	ylabel('Dimension 2');
	if dim == 3, zlabel('Dimension 3'), zticks(linspace(0, 1, 3)); end
	title('Spectral Embedding Visualization');
	legend(arrayfun(@num2str, unique_labels, 'UniformOutput', false), 'Location', 'east');
	axis equal;
	grid on;
	hold off;
end