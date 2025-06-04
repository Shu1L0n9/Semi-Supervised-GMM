function s = logsumexp(x, dim)
    % LOGSUMEXP Compute log(sum(exp(x), dim)) in a numerically stable way.
    % Default DIM = 1 (compute along columns).
    
    x = double(x);
    if nargin == 1
        % Determine the dimension to sum over
        dim = find(size(x)~=1,1);
        if isempty(dim)
            dim = 1;
        end
    else
        dim = double(dim);
    end
    
    % Subtract the maximum value along the specified dimension
    y = max(x, [], dim);
    x = bsxfun(@minus, x, y);
    s = y + log(sum(exp(x), dim));
    i = find(~isfinite(y));
    if ~isempty(i)
        s(i) = y(i);
    end
end