yum install mesa-libGL -y

conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia
conda install nvidiacub fvcore iopath pyembree cupy cython pip -c nvidia -c conda-forge -c fvcore -c iopath -c bottler -y
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y

pip3 install cupy-cuda11x -i https://pypi.tuna.tsinghua.edu.cn/simple

conda uninstall pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia

