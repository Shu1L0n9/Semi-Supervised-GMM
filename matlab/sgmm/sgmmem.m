function [mix, options, errlog] = sgmmem(mix, x_unlabeled, x_labeled, c_labeled, options)
% SGMMEM: EM algorithm for semi-supervised Gaussian mixture model
% Input:
%   mix: Initial GMM parameters
%   x_unlabeled: Unlabeled data
%   x_labeled: Labeled data
%   c_labeled: Class labels for labeled data
%   options: Training options
% Output:
%   mix: Updated GMM parameters
%   options: Training options containing the final error
%   errlog: Log of errors for each iteration

% Check if mix structure and data dimensions match
errstring = consist(mix, 'gmm', [double(x_unlabeled); double(x_labeled)]);
if ~isempty(errstring)
    error(errstring);
end

ndata_unlabeled = double(size(x_unlabeled, 1));
ndata_labeled = double(size(x_labeled, 1));
xdim = double(size(x_unlabeled, 2));

% Ensure labeled data provides class labels
if ~isequal(size(c_labeled), [ndata_labeled, 1])
    error('Class labels must be a column vector with the same length as x_labeled.');
end

% Initialize beta field in mix if not present
if ~isfield(mix, 'beta')
    num_classes = double(max(c_labeled));
    mix.beta = double(ones(num_classes, mix.ncentres) / mix.ncentres);
end

% Parse configuration options and set maximum number of iterations
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

% Perform E-step and M-step in the main loop
for n = 1:niters
    % E-step: Calculate posterior probabilities for unlabeled and labeled data
    % Log probability density for unlabeled data
    [post_unlabeled, log_act_unlabeled] = sgmmpost(mix, double(x_unlabeled));

    % Log probability density for labeled data
    log_post_labeled = double(zeros(ndata_labeled, mix.ncentres));
    for i = 1:ndata_labeled
        c = double(c_labeled(i));
        for l = 1:mix.ncentres
            beta_cl = double(mix.beta(c, l));
            log_pdf = gpdf(mix, double(x_labeled(i, :)), double(l));
            log_post_labeled(i, l) = log(double(mix.priors(l))) + log(beta_cl) + log_pdf;
        end
        % Normalize log posterior probabilities
        log_post_labeled(i, :) = log_post_labeled(i, :) - logsumexp(double(log_post_labeled(i, :)), 2);
    end

    % Convert to probability space
    post_labeled = exp(double(log_post_labeled));

    % Calculate the log-likelihood and check convergence conditions
    if (double(display) || store || test)
        % Log-likelihood for unlabeled data
        log_prob_unlabeled = logsumexp(double(log_act_unlabeled) + log(double(mix.priors)), 2);

        % Log-likelihood for labeled data
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

        % Negative log-likelihood
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

    % M-step: Update priors, means, and covariances
    N = double(ndata_unlabeled + ndata_labeled); % N: Total data size
    mix.priors = (double(sum(post_unlabeled,1)) + double(sum(post_labeled,1))) / N;
    % Corresponds to the formula for updating priors

    for l = 1:mix.ncentres
        numerator = double(x_unlabeled') * double(post_unlabeled(:,l)) + double(x_labeled') * double(post_labeled(:,l));
        denominator = N * double(mix.priors(l));
        mix.centres(l,:) = numerator / denominator;
        % Corresponds to the formula for updating means
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
                % Corresponds to the formula for updating spherical covariance
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
                % Corresponds to the formula for updating diagonal covariance
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
                % Corresponds to part of the formula for updating full covariance

                % Calculate weighted outer products for labeled data
                for i = 1:ndata_labeled
                    diff = (double(x_labeled(i,:)) - double(mix.centres(l,:)))';
                    cov_labeled = cov_labeled + double(post_labeled(i,l)) * (diff * diff');
                end
                cov_labeled = cov_labeled / (N * double(mix.priors(l)));
                % Corresponds to part of the formula for updating full covariance

                cov_combined = double(cov_unlabeled + cov_labeled);
                mix.covars(:,:,l) = double(cov_combined);
                % Corresponds to the formula for updating full covariance

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
    % Update beta
    for l = 1:mix.ncentres
        for c = 1:double(max(c_labeled))
            numerator = double(sum(post_labeled(c_labeled == c, l)));
            denominator = double(sum(post_labeled(:, l)));
            mix.beta(c, l) = numerator / denominator;
            % Corresponds to the formula for updating beta
        end
    end
end
% Set final options
options(8) = double(e);
if double(display) >= 0
    disp('Maximum number of iterations reached.');
end
end