function [predictions] = sgmmpred(mix, X)
% SGMM_PREDICT Predict class labels using a trained SGMM model.
%
%   Parameters:
%       mix: A trained SGMM model structure.
%       X: Data points matrix to predict (each row is a data point).
%
%   Returns:
%       predictions: Vector of predicted class labels.

fprintf('------------------------------\n');
fprintf('Predicting %d samples...\n', double(size(X, 1)));

% Compute posterior probabilities (similar to the E-step)
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
            % Retrieve PPCA parameters
            U = double(mix.U(:,:,l));        % Principal component matrix
            lambda = double(mix.lambda(l,:)); % Eigenvalues
            sigma2 = double(mix.covars(l));   % Noise variance
            diffs = double(X) - double(repmat(mix.centres(l,:), double(size(X,1)), 1));

            % Compute covariance matrix in reduced space
            M = double(diag(double(lambda))) + double(sigma2) * double(eye(double(size(U,2))));

            % Compute projection
            proj = double(diffs) * double(U);

            % Compute probability density
            inv_M = double(inv(double(M)));
            det_M = double(det(double(M)));
            exponent = double(-0.5) * double(sum((proj * inv_M) .* proj, 2));
            pdf = double(exp(exponent)) ./ double(sqrt((2*pi)^double(size(U,2)) * det_M));
        otherwise
            error(['Unknown covariance type ', double(mix.covar_type)]);
    end
    post(:, l) = double(double(mix.priors(l)) * double(pdf));
end

post = double(post) ./ double(sum(double(post), 2));

% Compute posterior probabilities for classes
class_posteriors = double(post) * double(mix.beta)';

% Predict class labels
[~, predictions] = max(double(class_posteriors), [], 2);

fprintf('Prediction complete!\n');
end