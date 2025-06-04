function [post, log_act] = sgmmpost(mix, x)
% SGMMPOST 计算后验概率和对数激活值

% 检查输入是否一致
errstring = consist(mix, 'gmm', double(x));
if ~isempty(errstring)
    error(errstring);
end

% 计算对数激活值
log_act = double(zeros(size(x, 1), mix.ncentres));
for k = 1:double(mix.ncentres)
    log_act(:, k) = gpdf(mix, double(x), double(k));
end

% 计算未归一化的对数后验概率
log_post_unnormalized = log(double(mix.priors)) + log_act;

% 归一化因子
log_sum_exp = logsumexp(log_post_unnormalized, 2);

% 计算对数后验概率
log_post = log_post_unnormalized - log_sum_exp;

% 转换为概率空间
post = exp(log_post);
end