CONDA_ENV=py310cu118
conda -V # check version
conda info --envs
#conda create -n $CONDA_ENV python=3.10 #tensorrt==8.5.3.1 does not support python3.11
#conda deactivate
conda activate $CONDA_ENV
conda info