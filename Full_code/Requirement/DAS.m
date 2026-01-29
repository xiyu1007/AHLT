function [anchors, anchor_idx] = DAS(X, m)
% DAS: 直接交替采样（Directly Alternate Sampling）锚点选择方法
%
% 输入:
%   X - 元胞数组，大小为 [1 x nV]，每个元胞是一个视图的数据矩阵，维度为 [n x d_v]
%       其中 n 是样本总数，d_v 是第 v 个视图的特征维度。
%   m - 要选择的锚点数量（建议 m >= 真实类别数）
%
% 输出:
%   anchor_idx    - 列向量，长度为 m，包含所选锚点的样本索引（从 1 到 n）
%   anchor_views  - 元胞数组，大小为 [1 x nV]，每个元胞是 [m x d_v] 的锚点特征矩阵，
%                   即从每个视图中提取出的 m 个锚点对应的特征
    % 步骤1: 拼接所有视图的特征
    X_concat = cat(2, X{:});  % [n x d_total]

    % 步骤2: 对每一维特征进行非负化（每列减去该列最小值）
    X_concat = X_concat - min(X_concat, [], 1);

    % 步骤3: 计算每个样本的初始得分：所有维度特征值之和
    s = sum(X_concat, 2);  % [n x 1]
    n = size(X_concat, 1);

    clear X_concat;

    anchor_idx = zeros(m, 1);        % 存储锚点索引
    selected = false(n, 1);          % 标记是否已被选为锚点

    % 步骤4: 迭代选择 m 个锚点
    for iter = 1:m
        % 将已选样本的得分设为负无穷，防止重复选择
        s(selected) = -inf;

        % 选择当前得分最高的样本
        [~, idx] = max(s);
        anchor_idx(iter) = idx;
        selected(idx) = true;

        % 若已选满 m 个，提前退出
        if iter == m
            break;
        end

        % 归一化得分：s = s / max(s)
        s_max = max(s);
        if s_max == 0
            % 安全处理：若剩余得分全为0，则随机选择剩余未选样本
            remaining = find(~selected);
            needed = m - iter;
            if needed > 0
                extra = datasample(remaining, min(needed, length(remaining)), 'Replace', false);
                anchor_idx(iter+1:end) = extra(:);
            end
            break;
        end
        s = s / s_max;

        % 更新得分：s = s .* (1 - s)
        % 目的：抑制刚选中的高分样本，提升中等分数样本的相对重要性，实现跨簇交替采样
        s = s .* (1 - s);
    end

    % 步骤5: 根据选中的索引，从每个视图中提取对应的锚点特征
    nV = length(X);  % 视图数量
    anchors = cell(1, nV);
    for v = 1:nV
        % X{v} 是 [n x d_v]，anchor_idx 是 [m x 1]，提取后为 [m x d_v]
        anchors{v} = X{v}(anchor_idx, :);
    end
end
