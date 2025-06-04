function s = logsumexp(x, dim)
    % LOGSUMEXP 计算 log(sum(exp(x),dim))，同时避免数值下溢。
    % 默认 DIM = 1（按列计算）。
    
    x = double(x);
    if nargin == 1
        % 确定求和的维度
        dim = find(size(x)~=1,1);
        if isempty(dim)
            dim = 1;
        end
    else
        dim = double(dim);
    end
    
    % 减去每个维度的最大值
    y = max(x, [], dim);
    x = bsxfun(@minus, x, y);
    s = y + log(sum(exp(x), dim));
    i = find(~isfinite(y));
    if ~isempty(i)
        s(i) = y(i);
    end
end