% filepath: /c:/Users/Shu1L/Desktop/sgmm_matlab/sgmm/sgmmactiv.m
function a = sgmmactiv(mix, x)
% SGMMACTIV 计算高斯混合模型的激活值
% 输入:
%   mix - 混合模型结构体
%   x   - 数据矩阵 (ndata x dim)
% 输出:
%   a   - 激活值矩阵 (ndata x ncentres)

% Check that inputs are consistent
errstring = consist(mix, 'gmm', double(x));
if ~isempty(errstring)
    error(errstring);
end

[ndata, dim] = size(double(x));
a = double(zeros(ndata, mix.ncentres));  % Preallocate matrix

switch mix.covar_type

    case 'spherical'
        % 对于球形协方差，直接使用广播
        wi2 = double(2) .* double(mix.covars')';  % 转置为1xncentres
        normal = double((2*pi)^(dim/2));
        normal = normal .* double(prod(sqrt(wi2)));

        % 计算数据点与中心点之间的距离
        distances = double(dist(double(x), double(mix.centres)));

        % 计算指数项
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
            % 使用协方差矩阵的Cholesky分解以加速计算
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