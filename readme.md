# Semi-Supervised Gaussian Mixture Model (SGMM) 双语言实现

本项目提供了**两种不同编程语言**的半监督高斯混合模型实现（MATLAB 和 Python），基于论文：

**"A Mixture of Experts Classifier with Learning Based on Both Labelled and Unlabelled Data"**  
*作者：David J. Miller and Hasan S. Uyar*

## 项目结构

```plaintext
Semi-Supervised GMM/
│
├── matlab/                         # MATLAB 实现版本
│   ├── test_sgmm.m                # SGMM 基础功能验证实验
│   ├── synthetic_data_classification.m # 合成数据分类实验
│   ├── gmm/                       # 标准 GMM 实现（用于对比和数据生成）
│   │   ├── gmm.m                  # GMM 核心函数
│   │   ├── gmmsamp.m              # GMM 采样函数
│   │   └── ···                    # 其他 GMM 相关函数
│   └── sgmm/                      # 半监督 GMM 实现
│       ├── sgmminit.m             # SGMM 初始化
│       ├── sgmmem.m               # SGMM EM 算法
│       ├── sgmmpred.m             # SGMM 预测
│       ├── sgmmpost.m             # SGMM 后验概率计算
│       └── ···                    # 其他 SGMM 相关函数
│
├── python/                         # Python 实现版本
│   ├── sgmm_core.py               # SGMM 核心算法实现
│   ├── train_sgmm.py              # SGMM 训练脚本
│   └── utils.py                   # 工具函数
│
├── data/                           # 数据集目录
│   ├── mnist/                     # MNIST 数据集（需要自己准备）
│   └── synthetic_data/            # 合成数据集
│       ├── synthetic_data_train.csv
│       └── synthetic_data_test.csv
│
└── README.md                      # 项目说明文档
```

## 两种实现版本

### MATLAB 版本
- **特点**：原始论文算法的直接实现，完全遵循论文中的数学公式
- **优势**：数学计算精确，矩阵运算高效，适合算法研究和验证
- **适用场景**：学术研究、算法验证、小规模实验

### Python 版本  
- **特点**：基于现代机器学习框架的优化实现
- **优势**：代码可读性强，易于扩展，支持大规模数据处理
- **适用场景**：实际应用、大规模实验、工程项目

## 算法简介

半监督高斯混合模型 (SGMM) 是一种混合专家分类器，能够同时利用有标签和无标签数据进行学习。该方法的核心思想是：

- **有标签数据**：用于监督学习，直接优化分类性能
- **无标签数据**：通过 EM 算法进行无监督学习，改善模型的密度估计
- **混合学习**：结合两种数据类型的信息，提高分类精度和泛化能力

支持三种协方差矩阵类型：
- `spherical`：球形协方差（各分量独立且方差相等）
- `diag`：对角协方差（各分量独立但方差不等）
- `full`：完全协方差（允许分量间相关）

## 实验脚本说明

### MATLAB 实验

#### 1. test_sgmm.m
基础功能验证实验。使用 GMM 生成简单的合成数据集，验证 SGMM 核心函数（`sgmminit`、`sgmmem` 等）的正确性。支持三种协方差类型的测试。

#### 2. synthetic_data_classification.m
合成数据分类实验。对 dollar sign 形状的合成数据进行分类，主要评估使用完全协方差矩阵时 SGMM 的分类性能。

### Python 实验

#### 1. train_sgmm.py
SGMM 训练脚本，提供完整的训练流程和参数设置。

#### 2. sgmm_core.py
包含 SGMM 的核心算法实现，包括初始化、EM 算法更新和预测功能。

## 快速开始

### MATLAB 版本

#### 环境要求
- MATLAB R2016b 或更高版本

#### 运行步骤
1. 在 MATLAB 中打开项目目录
2. 添加子文件夹到 MATLAB 路径：

```matlab
addpath('matlab/sgmm');
addpath('matlab/gmm');
addpath('data');
```

3. 运行基础验证实验：

```matlab
cd matlab
test_sgmm
```

4. 运行合成数据分类实验：

```matlab
synthetic_data_classification
```

### Python 版本

#### 环境要求
- Python 3.7 或更高版本
- NumPy, SciPy, scikit-learn, matplotlib 等依赖包

#### 运行步骤
1. 安装依赖包：

```bash
pip install numpy scipy scikit-learn matplotlib pandas
```

2. 运行训练脚本：

```bash
cd python
python train_sgmm.py
```

## 参考文献

Miller, David J, and Hasan Uyar. “A Mixture of Experts Classifier with Learning Based on Both Labelled and Unlabelled Data.” In Advances in Neural Information Processing Systems, Vol. 9. MIT Press, 1996.

## 许可证

本项目仅供学术研究使用。