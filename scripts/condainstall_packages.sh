# CONDA_ENV=py310cu118
# conda -V # check version
# conda info --envs
# conda activate $CONDA_ENV

# Verify Tensorflow installation
python3 -c "import tensorflow as tf; print('tf version:', tf.__version__); print(tf.config.list_physical_devices('GPU'))"

# Verify Pytorch installation
python3 -c "import torch; print('Torch version:', torch.__version__); print(torch.cuda.is_available())"

python3 -c "import tensorrt; print('Tensorrt version:', tensorrt.__version__)"

conda install -y -c conda-forge jupyterlab
conda install -y ipykernel
jupyter kernelspec list #view current jupyter kernels
ipython kernel install --user --name=$CONDA_ENV

conda install -y -c intel scikit-learn
conda install -y scikit-learn-intelex
conda install -y numpy matplotlib pandas Pillow scipy pyyaml scikit-image 

pip install -q torchinfo
pip install pynvml #see more GPU information
pip install onnx onnx-simplifier onnxruntime seaborn


pip install pyqt5
pip install pyqt6
pip install PySide6
pip install pyqtgraph
pip install opencv-python-headless
pip install PyOpenGL PyOpenGL_accelerate pyopengl

pip install sionna DeepMIMO
pip install pyadi-iio
pip install pygame

pip install -U --trusted-host www.open3d.org -f http://www.open3d.org/docs/latest/getting_started.html open3d
# Verify installation
python -c "import open3d as o3d; print(o3d.__version__)"

conda install -y -c conda-forge accelerate
conda install -y -c huggingface transformers


