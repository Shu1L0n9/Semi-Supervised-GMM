function [mix, options, errlog] = sgmmem(mix, x_unlabeled, x_labeled, c_labeled, options)
% SGMMEM: 通过EM算法进行半监督高斯混合模型的训练
% 输入:
%   mix: 初始GMM参数
%   x_unlabeled: 无标签数据
%   x_labeled: 有标签数据
%   c_labeled: 有标签数据的类别标签
%   options: 训练选项
% 输出:
%   mix: 更新后的GMM参数
%   options: 包含最终误差的训练选项
%   errlog: 记录每次迭代的误差

% 检查mix结构和数据尺寸是否匹配
errstring = consist(mix, 'gmm', [double(x_unlabeled); double(x_labeled)]);
if ~isempty(errstring)
    error(errstring);
end

ndata_unlabeled = double(size(x_unlabeled, 1));
ndata_labeled = double(size(x_labeled, 1));
xdim = double(size(x_unlabeled, 2));

% 确保有标签数据提供了类别标签
if ~isequal(size(c_labeled), [ndata_labeled, 1])
    error('类别标签必须是与 x_labeled 长度相同的列向量。');
end

% 如果mix中无beta字段则初始化
if ~isfield(mix, 'beta')
    num_classes = double(max(c_labeled));
    mix.beta = double(ones(num_classes, mix.ncentres) / mix.ncentres);
end

% 解析配置选项并设置最大迭代次数
if options(14)
    niters = double(options(14));
else
    niters = 100;
end

display = double(options(1));
store = 0;
if (nargout > 2)
    store = 1;
    errlog = double(zeros(1, niters));
end
test = 0;
if double(options(3)) > 0.0
    test = 1;
end

check_covars = 0;
if double(options(5)) >= 1
    if double(display) >= 0
        disp('check_covars is on');
    end
    check_covars = 1;
    MIN_COVAR = double(eps);
    init_covars = double(mix.covars);
end

% 在主循环中执行E步和M步
for n = 1:niters
    % E步: 计算无标签和有标签数据的后验概率
    % 无标签数据的对数概率密度
    [post_unlabeled, log_act_unlabeled] = sgmmpost(mix, double(x_unlabeled));

    % 有标签数据的对数概率密度
    log_post_labeled = double(zeros(ndata_labeled, mix.ncentres));
    for i = 1:ndata_labeled
        c = double(c_labeled(i));
        for l = 1:mix.ncentres
            beta_cl = double(mix.beta(c, l));
            log_pdf = gpdf(mix, double(x_labeled(i, :)), double(l));
            log_post_labeled(i, l) = log(double(mix.priors(l))) + log(beta_cl) + log_pdf;
        end
        % 归一化对数后验概率
        log_post_labeled(i, :) = log_post_labeled(i, :) - logsumexp(double(log_post_labeled(i, :)), 2);
    end

    % 转换为概率空间
    post_labeled = exp(double(log_post_labeled));

    % 计算负对数似然并检查收敛条件
    if (double(display) || store || test)
        % 无标签数据的对数似然
        log_prob_unlabeled = logsumexp(double(log_act_unlabeled) + log(double(mix.priors)), 2);

        % 有标签数据的对数似然
        log_prob_labeled = double(zeros(ndata_labeled, 1));
        for i = 1:ndata_labeled
            c = double(c_labeled(i));
            log_prob = -inf;
            for l = 1:mix.ncentres
                beta_cl = double(mix.beta(c, l));
                log_pdf = gpdf(mix, double(x_labeled(i, :)), double(l));
                log_prob = logsumexp([double(log_prob), log(double(mix.priors(l))) + log(beta_cl) + log_pdf], 2);
            end
            log_prob_labeled(i) = log_prob;
        end

        % 负对数似然
        log_prob = [double(log_prob_unlabeled); double(log_prob_labeled)];
        e = -double(sum(log_prob));
        if store
            errlog(n) = e;
        end
        if double(display) > 0
            fprintf(1, 'Cycle %4d  Error %11.6f\n', double(n), e);
        end
        if test
            if (double(n) > 1 && abs(e - eold) < double(options(3)))
                options(8) = e;
                return;
            else
                eold = e;
            end
        end
    end

    % M步: 更新先验概率、均值和协方差
    N = double(ndata_unlabeled + ndata_labeled); % N: 总数据量
    mix.priors = (double(sum(post_unlabeled,1)) + double(sum(post_labeled,1))) / N;
    % 对应公式中的 \alpha_l^{t+1} = \frac{1}{N} \left( \sum_{x_i \in \mathcal{X}_l} P(l|x_i, c_i, \Theta^t) + \sum_{x_i \in \mathcal{X}_u} P(l|x_i, \Theta^t) \right)

    for l = 1:mix.ncentres
        numerator = double(x_unlabeled') * double(post_unlabeled(:,l)) + double(x_labeled') * double(post_labeled(:,l));
        denominator = N * double(mix.priors(l));
        mix.centres(l,:) = numerator / denominator;
        % 对应公式中的 \mu_l^{t+1} = \frac{\sum_{x_i \in \mathcal{X}_l} x_i P(l|x_i, c_i, \Theta^t) + \sum_{x_i \in \mathcal{X}_u} x_i P(l|x_i, \Theta^t)}{N \alpha_l^t}
    end

    switch mix.covar_type
        case 'spherical'
            for l = 1:mix.ncentres
                n2_unlabeled = dist2(double(x_unlabeled), double(mix.centres(l,:)));
                n2_labeled = dist2(double(x_labeled), double(mix.centres(l,:)));
                v_unlabeled = double(post_unlabeled(:, l))' * double(n2_unlabeled);
                v_labeled = double(post_labeled(:, l))' * double(n2_labeled);
                v = v_unlabeled + v_labeled;
                mix.covars(l) = v / (N * double(mix.priors(l)) * xdim);
                % 对应球形协方差的更新公式: \sigma_l^{2, t+1} = \frac{\sum_{x_i \in \mathcal{X}_u} P(l|x_i, \Theta^t) ||x_i - \mu_l^t||^2 + \sum_{x_i \in \mathcal{X}_l} P(l|x_i, c_i, \Theta^t) ||x_i - \mu_l^t||^2}{N \alpha_l^t D}
                if check_covars
                    if double(mix.covars(l)) < MIN_COVAR
                        mix.covars(l) = init_covars(l);
                    end
                end
            end
        case 'diag'
            for l = 1:mix.ncentres
                diffs_unlabeled = double(x_unlabeled) - double(repmat(mix.centres(l,:), ndata_unlabeled, 1));
                cov_unlabeled = diffs_unlabeled' * (diffs_unlabeled .* double(repmat(post_unlabeled(:, l), 1, xdim))) / (N * double(mix.priors(l)));
                diffs_labeled = double(x_labeled) - double(repmat(mix.centres(l,:), ndata_labeled, 1));
                cov_labeled = diffs_labeled' * (diffs_labeled .* double(repmat(post_labeled(:, l), 1, xdim))) / (N * double(mix.priors(l)));
                cov_combined = double(cov_unlabeled + cov_labeled);
                mix.covars(l,:) = diag(double(cov_combined));
                % 对应对角协方差的更新公式，对角线上的元素 \Sigma_{l,jj}^{t+1} = \frac{\sum_{x_i \in \mathcal{X}_u} P(l|x_i, \Theta^t) (x_{ij} - \mu_{lj})^2 + \sum_{x_i \in \mathcal{X}_l} P(l|x_i, c_i, \Theta^t) (x_{ij} - \mu_{lj})^2}{N \alpha_l^t}
                if check_covars
                    if any(double(mix.covars(l,:)) < MIN_COVAR)
                        mix.covars(l,:) = init_covars(l,:);
                    end
                end
            end
        case 'full'
            % Corrected 'full' covariance update
            for l = 1:mix.ncentres
                % Initialize covariance matrix
                cov_unlabeled = zeros(xdim, xdim);
                cov_labeled = zeros(xdim, xdim);

                % Calculate weighted outer products for unlabeled data
                for i = 1:ndata_unlabeled
                    diff = (double(x_unlabeled(i,:)) - double(mix.centres(l,:)))';
                    cov_unlabeled = cov_unlabeled + double(post_unlabeled(i,l)) * (diff * diff');
                end
                cov_unlabeled = cov_unlabeled / (N * double(mix.priors(l)));
                % 对应公式中的一部分: \frac{1}{N \alpha_l^t} \sum_{x_i \in \mathcal{X}_u} P(l|x_i, \Theta^t) (x_i - \mu_l^t)(x_i - \mu_l^t)^T

                % Calculate weighted outer products for labeled data
                for i = 1:ndata_labeled
                    diff = (double(x_labeled(i,:)) - double(mix.centres(l,:)))';
                    cov_labeled = cov_labeled + double(post_labeled(i,l)) * (diff * diff');
                end
                cov_labeled = cov_labeled / (N * double(mix.priors(l)));
                % 对应公式中的一部分: \frac{1}{N \alpha_l^t} \sum_{x_i \in \mathcal{X}_l} P(l|x_i, c_i, \Theta^t) (x_i - \mu_l^t)(x_i - \mu_l^t)^T

                cov_combined = double(cov_unlabeled + cov_labeled);
                mix.covars(:,:,l) = double(cov_combined);
                % 对应公式中的 \Sigma_l^{t+1} = \frac{1}{N\alpha_l^t} \left( \sum_{x_i \in \mathcal{X}_l} P(l|x_i, c_i, \Theta^t) (x_i - \mu_l^t)(x_i - \mu_l^t)^T + \sum_{x_i \in \mathcal{X}_u} P(l|x_i, \Theta^t) (x_i - \mu_l^t)(x_i - \mu_l^t)^T \right)

                if check_covars
                    [~, S, ~] = svd(double(mix.covars(:,:,l)));
                    if min(double(S)) < MIN_COVAR
                        mix.covars(:,:,l) = init_covars(:,:,l);
                    end
                end
            end
        case 'ppca'
            for l = 1:mix.ncentres
                diffs_unlabeled = double(x_unlabeled) - double(repmat(mix.centres(l,:), ndata_unlabeled, 1));
                diffs_labeled = double(x_labeled) - double(repmat(mix.centres(l,:), ndata_labeled, 1));

                weighted_diffs_unlabeled = bsxfun(@times, post_unlabeled(:,l), diffs_unlabeled);
                weighted_diffs_labeled = bsxfun(@times, post_labeled(:,l), diffs_labeled);

                cov_unlabeled = diffs_unlabeled' * weighted_diffs_unlabeled / (N * double(mix.priors(l)));
                cov_labeled = diffs_labeled' * weighted_diffs_labeled / (N * double(mix.priors(l)));

                cov_combined = double(cov_unlabeled + cov_labeled);

                [tempcovars, tempU, templambda] = ppca(double(cov_combined), double(mix.ppca_dim));
                mix.covars(l) = double(tempcovars);
                mix.U(:,:,l) = double(tempU);
                mix.lambda(l,:) = double(templambda);
                if check_covars
                    if double(mix.covars(l)) < MIN_COVAR
                        mix.covars(l) = init_covars(l);
                    end
                end
            end
        otherwise
            error(['Unknown covariance type ', mix.covar_type]);
    end
    % 更新beta
    for l = 1:mix.ncentres
        for c = 1:double(max(c_labeled))
            numerator = double(sum(post_labeled(c_labeled == c, l)));
            denominator = double(sum(post_labeled(:, l)));
            mix.beta(c, l) = numerator / denominator;
            % 对应公式中的 \beta_{k|l}^{t+1} = \frac{\sum_{x_i \in \mathcal{X}_l, c_i=k} P(l|x_i, c_i, \Theta^t)}{\sum_{x_i \in \mathcal{X}_l} P(l|x_i, c_i, \Theta^t)}
        end
    end
end
% 设置最终选项
options(8) = double(e);
if double(display) >= 0
    disp('达到最大迭代次数。');
end
end