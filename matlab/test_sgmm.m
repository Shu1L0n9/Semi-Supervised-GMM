% test_sgmm.m
%
%   Experiment: Semi-supervised GMM on synthetic data
%   Experiment uses Gaussian Mixture Model (GMM) to generate synthetic datasets, and trains and evaluates the data based on Semi-supervised Gaussian Mixture Model (SGMM).
%   The experimental data is very simple, aiming to verify the correctness of functions such as sgmminit and sgmmem.
%
%   Covariance can be chosen from spherical, diag, full types.
%
%   Gaussian Mixture Model (GMM) code is from the Pattern Analysis Toolbox Netlab written by Ian Nabney.
%   https://www.mathworks.com/matlabcentral/fileexchange/2654-netlab
%
%

% Clear workspace and command window
clear;
clc;

% Add subfolders to path
addpath('gmm');
addpath('sgmm');

% Generate synthetic data
% Set basic parameters
dim = 2;                % Data dimension
ncentres = 9;           % Number of Gaussian components
ndata_total = 1800;     % Total number of data points
labeled_ratio = 0.01;   % Ratio of labeled data
covar_type = 'full';    % Covariance type: 'spherical', 'diag', 'full'

% Generate true Gaussian Mixture Model
true_mix = gmm(dim, ncentres, covar_type);

% Set covariance matrix based on covariance type
switch covar_type
    case 'spherical'
        true_mix.covars = 0.01 * ones(ncentres, 1);
    case 'diag'
        true_mix.covars = 0.01 * ones(ncentres, dim);
    case 'full'
        true_mix.covars = zeros(dim, dim, ncentres);
        for i = 1:ncentres
            true_mix.covars(:,:,i) = 0.01 * eye(dim);
        end
end

% Set centers
true_mix.centres = [
    1 1;
    -1 -1;
    1 -1;
    -1 1;
    2 2;
    -2 -2;
    2 -2;
    -2 2;
    0 0;
];

% Set prior probabilities
true_mix.priors = [1/9 1/9 1/9 1/9 1/9 1/9 1/9 1/9 1/9];

% Sample data from the true model
[data, labels] = gmmsamp(true_mix, ndata_total);

% Divide labeled and unlabeled data
% Apply labeled_ratio
ndata_labeled = round(ndata_total * labeled_ratio);
ndata_unlabeled = ndata_total - ndata_labeled;

% Randomly select indices for labeled data
labeled_idx = randperm(ndata_total, ndata_labeled);
unlabeled_idx = setdiff(1:ndata_total, labeled_idx);

% Separate data
x_labeled = data(labeled_idx, :);
c_labeled = labels(labeled_idx);
x_unlabeled = data(unlabeled_idx, :);
c_unlabeled = labels(unlabeled_idx);

% Initialize SGMM
mix = gmm(dim, ncentres, covar_type);
options = foptions;
options(1) = 1;     % Display iteration information
options(3) = 1e-6;  % Set convergence threshold
options(5) = 1;     % Set matrix check
options(14) = 100;  % Maximum number of iterations

% Initialize using kmeans
mix = sgmminit(mix, [x_unlabeled; x_labeled]);

% Print initialized SGMM parameters
disp('Initial SGMM parameters:');
disp(mix);

% Train SGMM
[mix, options, errlog] = sgmmem(mix, x_unlabeled, x_labeled, c_labeled, options);

% Predict using the trained model
predictions = sgmmpred(mix, data);  % Predict all data

% Visualize results
figure;

% Plot data points and prediction results
subplot(2,1,1);
hold on;

% Plot prediction results for all data, avoiding duplicate plotting
for i = 1:ncentres
    scatter(data(predictions==i,1), data(predictions==i,2), 20, 'o', 'DisplayName', sprintf('Predicted Class %d', i),'MarkerEdgeAlpha',0.3, 'MarkerFaceColor', 'none');
end

plot(mix.centres(:,1), mix.centres(:,2), 'kx', 'MarkerSize', 10, 'LineWidth', 2, ...
    'DisplayName', 'Learned Centers');
title('Data distribution, prediction results and learned centers');
legend('Location', 'best');
axis equal;
hold off;

% Plot error curve
subplot(2,1,2);
plot(errlog, 'b-', 'LineWidth', 1.5);
title('Training error curve');
xlabel('Number of iterations');
ylabel('Negative log-likelihood');
grid on;

% Print results
fprintf('\nTraining results:\n');
fprintf('Final error value: %.4f\n', options(8));
fprintf('\nLearned model parameters:\n');
fprintf('Centers:\n');
disp(mix.centres);
fprintf('Covariance:\n');
disp(mix.covars);
fprintf('Mixing coefficients:\n');
disp(mix.priors);