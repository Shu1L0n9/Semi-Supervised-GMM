function mix = sgmminit(mix, data)
% Get data dimensions and number of samples
[ndata, ndim] = size(data);

% Check that inputs are consistent
errstring = consist(mix, 'gmm', data);
if ~isempty(errstring)
    error(errstring);
end


% Run kmeans four additional times and keep the best clustering
best_dist = inf;
for i = 1:2
    [current_idx, current_centres, current_dist] = kmeans(data, mix.ncentres, ...
        'MaxIter', 10000, ...
        'Start', 'plus', ...
        'Display', 'off', ...
        'EmptyAction', 'error' ...
        );
    if current_dist < best_dist
        idx = current_idx;
        mix.centres = current_centres;
        best_dist = current_dist;
    end
end

% Ensure idx is a column vector
idx = idx(:);

% Create posterior probability matrix
post = zeros(size(data, 1), mix.ncentres);
indices = (1:size(data,1))';
post(sub2ind(size(post), indices, idx)) = 1;

% Set priors depending on number of points in each cluster
cluster_sizes = max(sum(post, 1), 1);  % Make sure that no prior is zero
mix.priors = cluster_sizes/sum(cluster_sizes); % Normalise priors

% Initialize covariance matrices according to covariance type
switch mix.covar_type
    case 'spherical'
        mix.covars = double(zeros(mix.ncentres, 1));
        for j = 1:mix.ncentres
            cluster_data = double(data(idx == j, :));
            if size(cluster_data, 1) > 1
                v = double(mean(var(cluster_data)));
                mix.covars(j) = double(v) + double(1e-5);
            else
                mix.covars(j) = double(1e-5);
            end
        end

    case 'diag'
        mix.covars = double(zeros(mix.ncentres, ndim));
        for j = 1:mix.ncentres
            cluster_data = double(data(idx == j, :));
            if size(cluster_data, 1) > 1
                v = double(var(cluster_data));
                mix.covars(j,:) = double(v) + double(1e-5);
            else
                mix.covars(j,:) = double(ones(1,ndim) * 1e-5);
            end
        end

    case 'full'
        mix.covars = double(zeros(ndim, ndim, mix.ncentres));
        for j = 1:mix.ncentres
            cluster_data = double(data(idx == j, :));
            if size(cluster_data, 1) > 1
                c = double(cov(cluster_data));
                mix.covars(:,:,j) = double(c) + double(eye(ndim) * 1e-5);
            else
                mix.covars(:,:,j) = double(eye(ndim) * 1e-5);
            end
        end

    case 'ppca'
        % Compute data variance for initialization
        data_var = double(var(data(:)));

        % Initialize PPCA-specific parameters
        mix.lambda = double(zeros(mix.ncentres, mix.ppca_dim));
        mix.U = double(zeros(mix.nin, mix.ppca_dim, mix.ncentres));
        mix.covars = double(zeros(mix.ncentres, 1));

        for i = 1:mix.ncentres
            % Initialize eigenvalues (lambda)
            mix.lambda(i,:) = double(ones(1, mix.ppca_dim) * data_var);

            % Initialize eigenvectors (U)
            [U, ~] = qr(randn(mix.nin, mix.ppca_dim), 0);
            mix.U(:,:,i) = double(U);

            % Initialize noise variance
            mix.covars(i) = double(data_var * 0.1);  % Set a small noise variance
        end
    otherwise
        error(['Unknown covariance type ', mix.covar_type]);
end