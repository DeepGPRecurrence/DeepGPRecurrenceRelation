# DeepGPRecurrenceRelation
Anonymized source code for paper "Characterizing Deep Gaussian Processes via Nonlinear Recurrence Relations"

## Requirements

The dependencies include PyTorch, GPyTorch, MPMath

PyTorch:
```
$ pip install torch==1.3.1+cu92 torchvision==0.4.2+cu92 -f https://download.pytorch.org/whl/torch_stable.html
```
GpyTorch:
```
$ pip install git+https://github.com/cornellius-gp/gpytorch.git@befb9961f5de7313b427202db1c79d744ce2bfde
```
MPMath to compute the geneneralized hypergeometric functions:
```
$ pip install mpmath
```
Or run the set up file which is located in ```bin/setup.sh``` if Conda is available in your machine

## Repository structure
This repository contains three main directories ```bin, data, figure, src```. Here, ```bin``` is to create new Conda environment and set up necessary packages. ```data``` contains most of saved models and generated data. ```figure``` contains all figures in the paper. ```src``` contains all the scripts for plotting and training models. At the beginning of any scripts, a short description is provided.

```
.
├── bin
├── data
│   ├── bifurcation
│   ├── dgp_regression_models_3
│   ├── dgp_regression_models_diabetes
│   ├── dgp_regression_models_share_kernel
│   ├── dgp_regression_models_share_kernel_diabetes
│   ├── exp_cosine_1d
│   ├── exp_per_1d
│   ├── exp_rq_1d
│   ├── exp_se_1d
│   ├── exp_se_high_dim
│   └── exp_sm_1d
├── figure
│   ├── bifurcation
│   ├── experiment
│   ├── experiment_boston
│   │   ├── no_sharing
│   │   └── sharing
│   ├── experiment_diabetes
│   │   ├── no_sharing
│   │   └── sharing
│   ├── fixed_points
│   └── track_expectation
└── src
```

