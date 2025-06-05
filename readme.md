# Semi-Supervised Gaussian Mixture Model (SGMM) Implementation

This project provides unofficial implementations of Semi-Supervised Gaussian from References.

## References

Miller, David J, and Hasan Uyar. “A Mixture of Experts Classifier with Learning Based on Both Labelled and Unlabelled Data.” In Advances in Neural Information Processing Systems, Vol. 9. MIT Press, 1996.

## 📚 Citation

You can cite the SGMM method proposed by Miller and Uyar using the following BibTeX entry:

```bibtex
@inproceedings{miller1996mixture,
  title     = {A mixture of experts classifier with learning based on both labelled and unlabelled data},
  author    = {Miller, David J. and Uyar, Hasan},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {9},
  year      = {1996}
}
```

## Project Structure

```plaintext
Semi-Supervised GMM/
│
├── matlab/                         # MATLAB implementation
│   ├── test_sgmm.m                # SGMM basic functionality test
│   ├── synthetic_data_classification.m # Synthetic data classification experiment
│   ├── gmm/                       # Standard GMM implementation (for comparison and data generation)
│   │   ├── gmm.m                  # GMM core functions
│   │   ├── gmmsamp.m              # GMM sampling functions
│   │   └── ···                    # Other GMM related functions
│   └── sgmm/                      # Semi-supervised GMM implementation
│       ├── sgmminit.m             # SGMM initialization
│       ├── sgmmem.m               # SGMM EM algorithm
│       ├── sgmmpred.m             # SGMM prediction
│       ├── sgmmpost.m             # SGMM posterior probability calculation
│       └── ···                    # Other SGMM related functions
│
├── python/                         # Python implementation
│   ├── sgmm_core.py               # SGMM core algorithm implementation
│   ├── train_sgmm.py              # SGMM training script
│   └── utils.py                   # Utility functions
│
├── data/                           # Dataset directory
│   ├── mnist/                     # MNIST dataset (prepare yourself)
│   └── synthetic_data/            # Synthetic dataset
│       ├── synthetic_data_train.csv
│       └── synthetic_data_test.csv
│
└── README.md                      # Project documentation
```

## Two Implementation Versions

### MATLAB Version
- **Features**: Direct implementation of the original paper algorithm, fully following the mathematical formulas in the paper
- **Advantages**: Precise mathematical calculations, efficient matrix operations, suitable for algorithm research and validation
- **Use Cases**: Academic research, algorithm validation, small-scale experiments

### Python Version  
- **Features**: Optimized implementation based on modern machine learning frameworks
- **Advantages**: High code readability, easy to extend, supports large-scale data processing
- **Use Cases**: Practical applications, large-scale experiments, engineering projects

## Algorithm Overview

The Semi-Supervised Gaussian Mixture Model (SGMM) is a mixture of experts classifier that can learn from both labeled and unlabeled data simultaneously. The core idea of this method is:

- **Labeled Data**: Used for supervised learning to directly optimize classification performance
- **Unlabeled Data**: Used for unsupervised learning through EM algorithm to improve model density estimation
- **Hybrid Learning**: Combines information from both data types to improve classification accuracy and generalization ability

Supports three types of covariance matrices:
- `spherical`: Spherical covariance (independent components with equal variance)
- `diag`: Diagonal covariance (independent components with unequal variance)
- `full`: Full covariance (allows correlation between components)

## Experiment Scripts

### MATLAB Experiments

#### 1. test_sgmm.m
Basic functionality verification experiment. Uses GMM to generate simple synthetic datasets to validate the correctness of SGMM core functions (`sgmminit`, `sgmmem`, etc.). Supports testing of three covariance types.

#### 2. synthetic_data_classification.m
Synthetic data classification experiment. Performs classification on dollar sign-shaped synthetic data, primarily evaluating SGMM classification performance using full covariance matrices.

### Python Experiments

#### 1. train_sgmm.py
SGMM training script providing complete training process and parameter settings.

#### 2. sgmm_core.py
Contains core SGMM algorithm implementation, including initialization, EM algorithm updates, and prediction functions.

## Quick Start

### MATLAB Version

#### Requirements
- MATLAB R2016b or higher

#### Steps
1. Open the project directory in MATLAB
2. Add subfolders to MATLAB path:

```matlab
addpath('matlab/sgmm');
addpath('matlab/gmm');
addpath('data');
```

3. Run basic verification experiment:

```matlab
cd matlab
test_sgmm
```

4. Run synthetic data classification experiment:

```matlab
synthetic_data_classification
```

### Python Version

#### Requirements
- Python 3.7 or higher
- Dependencies: NumPy, SciPy, scikit-learn, matplotlib, etc.

#### Steps
1. Install dependencies:

```bash
pip install numpy scipy scikit-learn matplotlib pandas
```

2. Run training script:

```bash
cd python
python train_sgmm.py
```

## License

This project is for academic research use only.