# #!/bin/bash

conda create -n dnalongbench python=3.9
conda activate dnalongbench
pip install selene-sdk==0.5.3

pip install torchmetrics kipoiseq==0.5.2 BioPython pandas Cython

pip install scipy==1.12.0 matplotlib==3.8 pyBigWig
pip install tensorflow==2.17.0
pip install typing_extensions==4.12.2
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 torchtext==0.16.0 -f https://download.pytorch.org/whl/cu118/torch_stable.html
pip install numpy==1.26.4 pandas==2.1.4
pip install natsort pytabix tqdm 

