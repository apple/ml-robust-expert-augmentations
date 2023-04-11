#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

echo "Running setup"
conda create -n ham python=3.9
source activate ham
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install git+https://github.com/diffeqml/torchdyn
pip install pandas
pip install jupyter
pip install --upgrade turibolt --index https://pypi.apple.com/simple
pip install turitrove -i https://pypi.apple.com/simple --upgrade
pip install markupsafe==2.0
pip install jinja2==3.0.3
export PYTHONPATH=.
which python
python -V
python3 -V
cp sensitivity.py /miniconda/envs/ham/lib/python3.9/site-packages/torchdyn/numerics/sensitivity.py
mkdir ham/data/autoencoder_aphynity/plots
echo "Setup step done"
