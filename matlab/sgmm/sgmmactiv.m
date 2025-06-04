% filepath: /c:/Users/Shu1L/Desktop/sgmm_matlab/sgmm/sgmmactiv.m
function a = sgmmactiv(mix, x)
% SGMMACTIV Compute activation values of Gaussian Mixture Model
% Input:
%   mix - mixture model structure
%   x   - data matrix (ndata x dim)
% Output:
%   a   - activation value matrix (ndata x ncentres)

% Check that inputs are consistent
errstring = consist(mix, 'gmm', double(x));
if ~isempty(errstring)
    error(errstring);
end

[ndata, dim] = size(double(x));
a = double(zeros(ndata, mix.ncentres));  % Preallocate matrix

switch mix.covar_type

    case 'spherical'
        % For spherical covariance, use broadcasting directly
        wi2 = double(2) .* double(mix.covars')';  % Transpose to 1 x ncentres
        normal = double((2*pi)^(dim/2));
        normal = normal .* double(prod(sqrt(wi2)));

        % Compute distances between data points and centers
        distances = double(dist(double(x), double(mix.centres)));

        % Compute exponential term
        a = double(exp(-distances ./ (wi2(ones(ndata, 1), :)))) ./ normal;

    case 'diag'
        normal = double((2*pi)^(mix.nin/2));
        s = double(prod(sqrt(mix.covars), 2));
        for j = 1:double(mix.ncentres)
            diffs = double(x) - (ones(double(ndata), 1) * double(mix.centres(j, :)));
            a(:, j) = double(exp(-0.5*sum((diffs .* diffs) ./ (ones(double(ndata), 1) * double(mix.covars(j, :))), 2))) ./ (normal .* s(j));
        end

    case 'full'
        normal = double((2*pi)^(mix.nin/2));
        for j = 1:double(mix.ncentres)
            diffs = double(x) - (ones(double(ndata), 1) * double(mix.centres(j, :)));
            % Use Cholesky decomposition of covariance matrix to speed up computation
            c = double(chol(double(mix.covars(:, :, j))));
            temp = double(diffs) / c;
            a(:, j) = double(exp(-0.5 * sum(temp .* temp, 2))) ./ (normal .* double(prod(diag(c))));
        end

    case 'ppca'
        log_normal = double(mix.nin) * double(log(2*pi));
        d2 = double(zeros(ndata, mix.ncentres));
        logZ = double(zeros(1, mix.ncentres));
        for i = 1:double(mix.ncentres)
            k = double(1 - mix.covars(i)) ./ double(mix.lambda(i, :));
            logZ(i) = double(log_normal) + double(mix.nin) * double(log(mix.covars(i))) - ...
                double(sum(log(1 - k)));
            diffs = double(x) - double(ones(ndata, 1) * double(mix.centres(i, :)));
            proj = double(diffs) * double(mix.U(:, :, i));
            d2(:, i) = (double(sum(diffs .* diffs, 2)) - ...
                double(sum((proj .* (ones(ndata, 1) * k)) .* proj, 2))) ./ ...
                double(mix.covars(i));
        end
        a = double(exp(-0.5 * (d2 + double(ones(ndata, 1) * logZ))));
    otherwise
        error(['Unknown covariance type ', mix.covar_type]);
end