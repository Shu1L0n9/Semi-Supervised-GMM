function [post, log_act] = sgmmpost(mix, x)
% SGMMPOST Calculate posterior probabilities and log activation values

% Check input consistency
errstring = consist(mix, 'gmm', double(x));
if ~isempty(errstring)
    error(errstring);
end

% Calculate log activation values
log_act = double(zeros(size(x, 1), mix.ncentres));
for k = 1:double(mix.ncentres)
    log_act(:, k) = gpdf(mix, double(x), double(k));
end

% Calculate unnormalized log posterior probabilities
log_post_unnormalized = log(double(mix.priors)) + log_act;

% Normalization factor
log_sum_exp = logsumexp(log_post_unnormalized, 2);

% Calculate log posterior probabilities
log_post = log_post_unnormalized - log_sum_exp;

% Convert to probability space
post = exp(log_post);
end