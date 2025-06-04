% synthetic_data_classification.m
%
%   实验使用半监督高斯混合模型 (SGMM) 对合成数据csv进行分类。
%   主要步骤包括数据加载、参数设置、数据划分、模型初始化与训练、模型评估以及结果可视化。
%   本实验旨在评估此半监督高斯混合模型使用 'full' 类型协方差矩阵时的性能。
%
%

% 清空工作区和命令行窗口
clear;
clc;

% 添加子文件夹到路径
addpath('sgmm');

% 读取训练和测试数据
fprintf('正在加载数据...\n');

% 读取训练数据
train_data_raw = readtable('data\synthetic_data\synthetic_data_train.csv');
train_data = table2array(train_data_raw(:, 1:3));
train_labels = table2array(train_data_raw(:, 4)) + 1;

% 读取测试数据
test_data_raw = readtable('data\synthetic_data\synthetic_data_test.csv');
test_data = table2array(test_data_raw(:, 1:3));
test_labels = table2array(test_data_raw(:, 4)) + 1;

% 设置参数
dim = 3;
labeled_ratio = 0.02;
ncentres = 40;

% 划分训练数据为有标签和无标签
ndata_total = size(train_data, 1);
ndata_labeled = round(ndata_total * labeled_ratio);
labeled_idx = randperm(ndata_total, ndata_labeled);
unlabeled_idx = setdiff(1:ndata_total, labeled_idx);

% 分离数据
x_labeled = train_data(labeled_idx, :);
c_labeled = train_labels(labeled_idx);
x_unlabeled = train_data(unlabeled_idx, :);
c_unlabeled = train_labels(unlabeled_idx);

% 初始化和训练 SGMM
fprintf('初始化SGMM模型...\n');
mix = gmm(dim, ncentres, 'full');
num_classes = max(c_labeled);

% 初始化 beta
mix.beta = rand(num_classes, ncentres);
mix.beta = normalise(mix.beta);

% 设置训练选项
options = foptions;
options(1) = 1;         % 显示迭代信息
options(3) = 0.0001;    % 设置收敛阈值
options(5) = 1;         % 设置矩阵检查
options(14) = 1000;     % 最大迭代次数

% 初始化模型参数
mix = sgmminit(mix, [x_labeled; x_unlabeled]);

% 训练模型
fprintf('训练 SGMM 模型...\n');
[mix, options, errlog] = sgmmem(mix, x_unlabeled, x_labeled, c_labeled, options);

% 评估模型
fprintf('\n模型评估:\n');

% 在训练集上评估
train_predictions = sgmmpred(mix, train_data);
train_accuracy = sum(train_predictions == train_labels) / length(train_labels);
fprintf('训练集准确率: %.2f%%\n', train_accuracy * 100);

% 在测试集上评估
test_predictions = sgmmpred(mix, test_data);
test_accuracy = sum(test_predictions == test_labels) / length(test_labels);
fprintf('测试集准确率: %.2f%%\n', test_accuracy * 100);

% 可视化结果
figure('Name', '训练集和测试集预测结果');

% 训练集可视化
subplot(2,1,1);
scatter3(train_data(:,1), train_data(:,2), train_data(:,3), 20, train_predictions, 'filled');
title('训练集预测结果');
xlabel('X'); ylabel('Y'); zlabel('Z');
grid on;

% 测试集可视化
subplot(2,1,2);
scatter3(test_data(:,1), test_data(:,2), test_data(:,3), 20, test_predictions, 'filled');
title('测试集预测结果');
xlabel('X'); ylabel('Y'); zlabel('Z');
grid on;

% 绘制训练误差曲线
figure('Name', '训练误差曲线');
plot(errlog, 'b-', 'LineWidth', 1.5);
title('SGMM训练误差曲线');
xlabel('迭代次数');
ylabel('负对数似然');
grid on;