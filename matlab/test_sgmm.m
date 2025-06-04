% test_sgmm.m
%
%   实验使用高斯混合模型 (GMM) 生成合成数据集，并基于半监督高斯混合模型 (SGMM) 对数据进行训练和评估。
%   实验数据非常简单，旨在验证 sgmminit 和 sgmmem 等函数的正确性。
%
%   协方差可以选择 spherical, diag, full 三种类型。
%
%   高斯混合模型 (GMM) 代码来自 Ian Nabney 编写的模式分析工具箱 Netlab。
%   https://www.mathworks.com/matlabcentral/fileexchange/2654-netlab
%
%

% 清空工作区和命令行窗口
clear;
clc;

% 添加子文件夹到路径
addpath('gmm');
addpath('sgmm');

% 生成合成数据
% 设置基本参数
dim = 2;                % 数据维度
ncentres = 9;           % 高斯分量数
ndata_total = 1800;     % 总数据点数
labeled_ratio = 0.01;   % 有标签数据的比例
covar_type = 'full';    % 协方差类型: 'spherical', 'diag', 'full'

% 生成真实的高斯混合模型
true_mix = gmm(dim, ncentres, covar_type);

% 根据协方差类型设置协方差矩阵
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

% 设置中心点
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

% 设置先验概率
true_mix.priors = [1/9 1/9 1/9 1/9 1/9 1/9 1/9 1/9 1/9];

% 从真实模型中采样数据
[data, labels] = gmmsamp(true_mix, ndata_total);

% 划分有标签和无标签数据
% 应用 labeled_ratio
ndata_labeled = round(ndata_total * labeled_ratio);
ndata_unlabeled = ndata_total - ndata_labeled;

% 随机选择有标签数据的索引
labeled_idx = randperm(ndata_total, ndata_labeled);
unlabeled_idx = setdiff(1:ndata_total, labeled_idx);

% 分离数据
x_labeled = data(labeled_idx, :);
c_labeled = labels(labeled_idx);
x_unlabeled = data(unlabeled_idx, :);
c_unlabeled = labels(unlabeled_idx);

% 初始化 SGMM
mix = gmm(dim, ncentres, covar_type);
options = foptions;
options(1) = 1;     % 显示迭代信息
options(3) = 1e-6;  % 设置收敛阈值
options(5) = 1;     % 设置矩阵检查
options(14) = 100;  % 最大迭代次数

% 使用 kmeans 初始化
mix = sgmminit(mix, [x_unlabeled; x_labeled]);

% 打印初始化的SGMM参数
disp('SGMM 初始参数：');
disp(mix);

% 训练 SGMM
[mix, options, errlog] = sgmmem(mix, x_unlabeled, x_labeled, c_labeled, options);

% 用训练好的模型进行预测
predictions = sgmmpred(mix, data);  % 预测所有数据

% 可视化结果
figure;

% 绘制数据点和预测结果
subplot(2,1,1);
hold on;

% 绘制所有数据的预测结果, 避免重复绘制
for i = 1:ncentres
    scatter(data(predictions==i,1), data(predictions==i,2), 20, 'o', 'DisplayName', sprintf('Predicted Class %d', i),'MarkerEdgeAlpha',0.3, 'MarkerFaceColor', 'none');
end

plot(mix.centres(:,1), mix.centres(:,2), 'kx', 'MarkerSize', 10, 'LineWidth', 2, ...
    'DisplayName', 'Learned Centers');
title('数据分布、预测结果和学习到的中心');
legend('Location', 'best');
axis equal;
hold off;

% 绘制误差曲线
subplot(2,1,2);
plot(errlog, 'b-', 'LineWidth', 1.5);
title('训练误差曲线');
xlabel('迭代次数');
ylabel('负对数似然');
grid on;

% 打印结果
fprintf('\n训练结果:\n');
fprintf('最终误差值: %.4f\n', options(8));
fprintf('\n学习到的模型参数:\n');
fprintf('中心点:\n');
disp(mix.centres);
fprintf('协方差:\n');
disp(mix.covars);
fprintf('混合系数:\n');
disp(mix.priors);