function [anchors, anchor_idx] = DAS(X, m)
    % DAS: Directly Alternate Sampling
    %
    % 输入:
    %   X - {1 x nV} cell，每个 X{v} 是 [n x d_v] 数据矩阵
    %   m - 要选择的锚点数
    %
    % 输出:
    %   anchors     - {1 x nV} cell，每个为 [m x d_v]
    %   anchor_idx  - [m x 1] 锚点索引
    nV = numel(X);
    n = size(X{1}, 1);

    % Step 1: 计算每个样本的加权得分 s_i = sum_v sum_d X{v}(i,d)
    s = zeros(n, 1, 'like', X{1});  % 保持数据类型一致
    for v = 1:nV
        Xv = X{v};
        % 中心化每个视图特征（按列减最小值）
        colMin = min(Xv, [], 1);
        Xv = bsxfun(@minus, Xv, colMin);
        % 累计特征和
        s = s + sum(Xv, 2);
        clear Xv colMin;
    end

    % Step 2: 选择 m 个锚点
    anchor_idx = zeros(m, 1);
    selected = false(n, 1);

    for iter = 1:m
        s(selected) = -inf;
        [~, idx] = max(s);
        anchor_idx(iter) = idx;
        selected(idx) = true;

        if iter == m
            break;
        end

        s_max = max(s);
        if s_max == 0
            remaining = find(~selected);
            needed = m - iter;
            if needed > 0
                extra = datasample(remaining, min(needed, length(remaining)), 'Replace', false);
                anchor_idx(iter+1:end) = extra(:);
            end
            break;
        end
        s = s / s_max;
        s = s .* (1 - s);
    end

    % Step 3: 提取锚点特征
    anchors = cell(1, nV);
    for v = 1:nV
        anchors{v} = X{v}(anchor_idx, :);
    end
end