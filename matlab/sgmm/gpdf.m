function log_pdf = gpdf(mix, x, k)
% 计算高斯分布的对数概率密度

mu = double(mix.centres(k, :));
xdim = double(length(double(x(1, :))));  % 数据维度
switch mix.covar_type
    case 'spherical'
        sigma2 = double(mix.covars(k));
        log_det = double(xdim) * double(log(double(sigma2)));
        inv_sigma = double(1) / double(sigma2);

        % 计算对数概率密度
        diff = double(x) - double(mu);  % (N x D)
        log_pdf = double(-0.5) * (double(xdim) * double(log(2*pi)) + double(log_det) + double(sum((diff .* diff) * double(inv_sigma), 2)));

    case 'diag'
        sigma2 = double(mix.covars(k, :));
        log_det = double(sum(double(log(sigma2))));
        inv_sigma = double(1) ./ double(sigma2);

        % 计算对数概率密度
        diff = double(x) - double(mu);  % (N x D)
        log_pdf = double(-0.5) * (double(xdim) * double(log(2*pi)) + double(log_det) + double(sum((diff .* diff) .* inv_sigma, 2)));

    case 'full'
        sigma = double(mix.covars(:, :, k));
        % 使用 Cholesky 分解计算对数行列式，提高数值稳定性
        [L, p] = chol(sigma, 'lower');
        if double(p) ~= 0
            error('协方差矩阵不是正定的.');
        end
        log_det = double(2) * sum(double(log(diag(L))));
        inv_sigma = double(inv(sigma));

        % 计算对数概率密度
        diff = double(x) - double(mu);  % (N x D)
        % 计算 (diff * inv_sigma) .* diff 并求和
        quad_form = double(sum((diff * inv_sigma) .* diff, 2));  % (N x 1)
        log_pdf = double(-0.5) * (double(xdim) * double(log(2*pi)) + double(log_det) + quad_form);

    case 'ppca'
        sigma2 = double(mix.covars(k));
        log_det = double(xdim) * double(log(sigma2));
        inv_sigma = double(1) / double(sigma2);

        % 计算对数概率密度
        diff = double(x) - double(mu);  % (N x D)
        log_pdf = double(-0.5) * (double(xdim) * double(log(2*pi)) + double(log_det) + double(sum((diff .* diff) * inv_sigma, 2)));

    otherwise
        error('未知的协方差类型');
end