#!/bin/bash

echo "Create a working environment and install necessary packages"

WORKINGENV='dgp'

conda create -y -n $WORKINGENV --clone root

source activate $WORKINGENV

pip install torch==1.3.1+cu92 torchvision==0.4.2+cu92 -f https://download.pytorch.org/whl/torch_stable.html

pip install git+https://github.com/cornellius-gp/gpytorch.git@befb9961f5de7313b427202db1c79d744ce2bfde

pip install mpmath

echo "Done!"
