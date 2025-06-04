function n2 = dist(x, c)
    % DIST Compute squared Euclidean distance between two sets of vectors
    %   n2 = dist(x, c) returns a matrix of squared Euclidean distances between
    %   each row vector in x and each row vector in c.
    x = double(x);
    c = double(c);
    [ndata, dimx] = size(x);
    [ncentres, dimc] = size(c);
    if dimx ~= dimc
        error('Data dimension does not match centre dimension');
    end
    
    n2 = (ones(ncentres, 1) * sum((x.^2)', 1))' + ...
        ones(ndata, 1) * sum((c.^2)',1) - ...
        2.*(x*(c'));
    
    % Rounding errors occasionally cause negative entries in n2
    if any(any(n2<0))
        n2(n2<0) = 0;
    end
end