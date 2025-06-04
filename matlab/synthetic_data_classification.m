% synthetic_data_classification.m
%
%   Experiment using Semi-supervised Gaussian Mixture Model (SGMM) to classify synthetic data csv.
%   Main steps include data loading, parameter setting, data splitting, model initialization and training, model evaluation, and result visualization.
%   This experiment aims to evaluate the performance of SGMM using 'full' type covariance matrix.
%
%

% Clear workspace and command window
clear;
clc;

% Add subfolders to path
addpath('sgmm');

% Load training and testing data
fprintf('Loading data...\n');

% Load training data
train_data_raw = readtable('data\synthetic_data\synthetic_data_train.csv');
train_data = table2array(train_data_raw(:, 1:3));
train_labels = table2array(train_data_raw(:, 4)) + 1;

% Load testing data
test_data_raw = readtable('data\synthetic_data\synthetic_data_test.csv');
test_data = table2array(test_data_raw(:, 1:3));
test_labels = table2array(test_data_raw(:, 4)) + 1;

% Set parameters
dim = 3;
labeled_ratio = 0.02;
ncentres = 40;

% Split training data into labeled and unlabeled
ndata_total = size(train_data, 1);
ndata_labeled = round(ndata_total * labeled_ratio);
labeled_idx = randperm(ndata_total, ndata_labeled);
unlabeled_idx = setdiff(1:ndata_total, labeled_idx);

% Separate data
x_labeled = train_data(labeled_idx, :);
c_labeled = train_labels(labeled_idx);
x_unlabeled = train_data(unlabeled_idx, :);
c_unlabeled = train_labels(unlabeled_idx);

% Initialize and train SGMM
fprintf('Initializing SGMM model...\n');
mix = gmm(dim, ncentres, 'full');
num_classes = max(c_labeled);

% Initialize beta
mix.beta = rand(num_classes, ncentres);
mix.beta = normalise(mix.beta);

% Set training options
options = foptions;
options(1) = 1;         % Display iteration information
options(3) = 0.0001;    % Set convergence threshold
options(5) = 1;         % Set matrix check
options(14) = 1000;     % Maximum number of iterations

% Initialize model parameters
mix = sgmminit(mix, [x_labeled; x_unlabeled]);

% Train model
fprintf('Training SGMM model...\n');
[mix, options, errlog] = sgmmem(mix, x_unlabeled, x_labeled, c_labeled, options);

% Evaluate model
fprintf('\nModel evaluation:\n');

% Evaluate on training set
train_predictions = sgmmpred(mix, train_data);
train_accuracy = sum(train_predictions == train_labels) / length(train_labels);
fprintf('Training accuracy: %.2f%%\n', train_accuracy * 100);

% Evaluate on testing set
test_predictions = sgmmpred(mix, test_data);
test_accuracy = sum(test_predictions == test_labels) / length(test_labels);
fprintf('Testing accuracy: %.2f%%\n', test_accuracy * 100);

% Visualize results
figure('Name', 'Training and Testing Predictions');

% Training set visualization
subplot(2,1,1);
scatter3(train_data(:,1), train_data(:,2), train_data(:,3), 20, train_predictions, 'filled');
title('Training Predictions');
xlabel('X'); ylabel('Y'); zlabel('Z');
grid on;

% Testing set visualization
subplot(2,1,2);
scatter3(test_data(:,1), test_data(:,2), test_data(:,3), 20, test_predictions, 'filled');
title('Testing Predictions');
xlabel('X'); ylabel('Y'); zlabel('Z');
grid on;

% Plot training error curve
figure('Name', 'Training Error Curve');
plot(errlog, 'b-', 'LineWidth', 1.5);
title('SGMM Training Error Curve');
xlabel('Iterations');
ylabel('Negative Log-Likelihood');
grid on;