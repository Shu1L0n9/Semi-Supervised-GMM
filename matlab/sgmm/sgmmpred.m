function [predictions] = sgmmpred(mix, X)
% SGMM_PREDICT 使用训练好的 SGMM 模型对新数据进行类别标签预测。
%
%   参数:
%       mix: 一个训练好的 SGMM 模型结构。
%       X: 要预测的数据点矩阵（每行是一个数据点）。
%
%   返回:
%       predictions: 预测的类别标签向量。

fprintf('------------------------------\n');
fprintf('预测 %d 个样本...\n', double(size(X, 1)));

% 计算后验概率（类似于 E 步骤）
post = double(zeros(double(size(X, 1)), double(mix.ncentres)));
for l = 1:double(mix.ncentres)
    switch mix.covar_type
        case 'spherical'
            diffs = double(X) - double(repmat(mix.centres(l,:), double(size(X,1)), 1));
            exponent = double(-0.5) * double(sum(diffs.^2, 2)) ./ double(mix.covars(l));
            pdf = double(exp(exponent)) ./ double(sqrt((2*pi*double(mix.covars(l)))^double(mix.nin)));
        case 'diag'
            diffs = double(X) - double(repmat(mix.centres(l,:), double(size(X,1)), 1));
            exponent = double(-0.5) * double(sum((diffs.^2) ./ double(mix.covars(l,:)), 2));
            pdf = double(exp(exponent)) ./ double(prod(sqrt(2*pi*double(mix.covars(l,:)))));
        case 'full'
            diffs = double(X) - double(repmat(mix.centres(l,:), double(size(X,1)), 1));
            inv_cov = double(inv(double(mix.covars(:,:,l))));
            det_cov = double(det(double(mix.covars(:,:,l))));
            exponent = double(-0.5) * double(sum((diffs * inv_cov) .* diffs, 2));
            pdf = double(exp(exponent)) ./ double(sqrt((2*pi)^double(mix.nin) * det_cov));
        case 'ppca'
            % 获取PPCA参数
            U = double(mix.U(:,:,l));        % 主成分矩阵
            lambda = double(mix.lambda(l,:)); % 特征值
            sigma2 = double(mix.covars(l));   % 噪声方差
            diffs = double(X) - double(repmat(mix.centres(l,:), double(size(X,1)), 1));

            % 计算降维空间的协方差矩阵
            M = double(diag(double(lambda))) + double(sigma2) * double(eye(double(size(U,2))));

            % 计算投影
            proj = double(diffs) * double(U);

            % 计算概率密度
            inv_M = double(inv(double(M)));
            det_M = double(det(double(M)));
            exponent = double(-0.5) * double(sum((proj * inv_M) .* proj, 2));
            pdf = double(exp(exponent)) ./ double(sqrt((2*pi)^double(size(U,2)) * det_M));
        otherwise
            error(['未知的协方差类型 ', double(mix.covar_type)]);
    end
    post(:, l) = double(double(mix.priors(l)) * double(pdf));
end

post = double(post) ./ double(sum(double(post), 2));

% 计算类别的后验概率
class_posteriors = double(post) * double(mix.beta)';

% 预测类别标签
[~, predictions] = max(double(class_posteriors), [], 2);

fprintf('预测完成！\n');
end