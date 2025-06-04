import numpy as np
import matplotlib.pyplot as plt
from sgmm_core import * 

# 加载保存的数据
loaded_data_train = np.loadtxt("data/synthetic_data/synthetic_data_train.csv", delimiter=",", skiprows=1)
X_train = loaded_data_train[:, :3]
y_train = loaded_data_train[:, 3].astype(int)

loaded_data_test = np.loadtxt("data/synthetic_data/synthetic_data_test.csv", delimiter=",", skiprows=1)
X_test = loaded_data_test[:, :3]
y_test = loaded_data_test[:, 3].astype(int)

# 打乱数据集
shuffle_idx_train = np.random.permutation(len(X_train))
X_train = X_train[shuffle_idx_train]
y_train = y_train[shuffle_idx_train]

shuffle_idx_test = np.random.permutation(len(X_test))
X_test = X_test[shuffle_idx_test]
y_test = y_test[shuffle_idx_test]

# 用训练集划分有标签和无标签数据
labeled_ratio = 0.01    # Labeled data ratio
labeled_indices = np.random.choice(X_train.shape[0], size=int(labeled_ratio * X_train.shape[0]), replace=False)
unlabeled_indices = np.setdiff1d(np.arange(X_train.shape[0]), labeled_indices)

X_labeled = X_train[labeled_indices]
y_labeled = y_train[labeled_indices]
X_unlabeled = X_train[unlabeled_indices]

# 使用训练集训练 SGMM 模型
n_components = 10
n_classes = 3
model = SemiSupervisedGMM(n_components=n_components, n_classes=n_classes, max_iter=5000, device='cpu')
model.fit(X_labeled, y_labeled, X_unlabeled)

# 对训练集和测试集进行预测
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# 评估模型
accuracy_train = np.mean(y_train_pred == y_train)
print(f"Accuracy on trainset: {accuracy_train:.4f}")
accuracy_test = np.mean(y_test_pred == y_test)
print(f"Accuracy on testset: {accuracy_test:.4f}")

# 只可视化测试集预测结果
fig = plt.figure(figsize=(10, 8), dpi=100)
ax = fig.add_subplot(111, projection='3d')

# 根据预测标签着色
colors = ['red', 'green', 'blue']
for i in range(len(X_test)):
    ax.scatter(X_test[i, 0], X_test[i, 1], X_test[i, 2], color=colors[y_test_pred[i]], s=1)

ax.view_init(elev=5, azim=-85)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title("Visualization of Point Clouds with Semi-Supervised GMM Predictions on Testset")

# 添加图例
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label='S-curve (Pred)', markerfacecolor='red', markersize=8),
                   plt.Line2D([0], [0], marker='o', color='w', label='Affine Plane (Pred)', markerfacecolor='green', markersize=8),
                   plt.Line2D([0], [0], marker='o', color='w', label='O-curve (Pred)', markerfacecolor='blue', markersize=8)]
ax.legend(handles=legend_elements, loc='upper right')

plt.show()