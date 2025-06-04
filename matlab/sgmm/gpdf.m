function log_pdf = gpdf(mix, x, k)
% Compute the log probability density of a Gaussian distribution

mu = double(mix.centres(k, :));
xdim = double(length(double(x(1, :))));  % Data dimension
switch mix.covar_type
    case 'spherical'
        sigma2 = double(mix.covars(k));
        log_det = double(xdim) * double(log(double(sigma2)));
        inv_sigma = double(1) / double(sigma2);

        % Compute log probability density
        diff = double(x) - double(mu);  % (N x D)
        log_pdf = double(-0.5) * (double(xdim) * double(log(2*pi)) + double(log_det) + double(sum((diff .* diff) * double(inv_sigma), 2)));

    case 'diag'
        sigma2 = double(mix.covars(k, :));
        log_det = double(sum(double(log(sigma2))));
        inv_sigma = double(1) ./ double(sigma2);

        % Compute log probability density
        diff = double(x) - double(mu);  % (N x D)
        log_pdf = double(-0.5) * (double(xdim) * double(log(2*pi)) + double(log_det) + double(sum((diff .* diff) .* inv_sigma, 2)));

    case 'full'
        sigma = double(mix.covars(:, :, k));
        % Use Cholesky decomposition to compute log determinant for numerical stability
        [L, p] = chol(sigma, 'lower');
        if double(p) ~= 0
            error('Covariance matrix is not positive definite.');
        end
        log_det = double(2) * sum(double(log(diag(L))));
        inv_sigma = double(inv(sigma));

        % Compute log probability density
        diff = double(x) - double(mu);  % (N x D)
        % Compute (diff * inv_sigma) .* diff and sum
        quad_form = double(sum((diff * inv_sigma) .* diff, 2));  % (N x 1)
        log_pdf = double(-0.5) * (double(xdim) * double(log(2*pi)) + double(log_det) + quad_form);

    case 'ppca'
        sigma2 = double(mix.covars(k));
        log_det = double(xdim) * double(log(sigma2));
        inv_sigma = double(1) / double(sigma2);

        % Compute log probability density
        diff = double(x) - double(mu);  % (N x D)
        log_pdf = double(-0.5) * (double(xdim) * double(log(2*pi)) + double(log_det) + double(sum((diff .* diff) * inv_sigma, 2)));

    otherwise
        error('Unknown covariance type');
end