#! /bin/bash

root=`pwd`

conda env create --file 1440Environment.yml
conda activate 1440Environment

# Install dependecies
conda install numpy matplotlib pillow scipy tqdm scikit-learn -y
pip install tensorflow-gpu==1.13.1
pip install tensorboardX==1.7

# install torchdiffeq
git clone https://github.com/rtqichen/torchdiffeq.git
cd torchdiffeq
pip install -e .
