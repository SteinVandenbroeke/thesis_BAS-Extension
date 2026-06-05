pip install uv

PIP_NO_CACHE_DIR=1
 
uv venv --python 3.9 .run_venv

source .run_venv/bin/activate

uv pip install --no-cache "numpy<1.24.0" #4.0.0

ln -s $(python -c "import numpy; print(numpy.get_include())")/numpy .run_venv/include/numpy


uv pip install --no-cache "setuptools<70" wheel

uv pip install --no-cache --no-build-isolation Cython

# Set environment variables to help the build find your CUDA

# Install dependencies needed to build MMCV

uv pip install --no-cache Cython "setuptools<70"

uv pip install imageio

uv pip install --no-cache torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install MMCV 2.x

uv pip install --no-cache mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html
python -c "import mmcv; from mmcv.ops import RoIAlign; print('MMCV Extension Loaded Successfully!')"

uv pip install --no-cache chainercv==0.13.1 --no-build-isolation
uv pip install --no-cache pycocotools scikit-image scipy tqdm
# Resume your project compilation

uv pip install --no-cache "numpy<1.24.0" #4.0.0

ln -s $(python -c "import numpy; print(numpy.get_include())")/numpy .run_venv/include/numpy

uv pip install "Cython<3.0.0" "numpy<2.0.0"

uv pip install git+https://github.com/lucasb-eyer/pydensecrf.git

# 2. Install pydensecrf and disable build isolation so it is forced to use the older Cython
uv pip install --no-build-isolation pydensecrf==1.0rc3

mkdir -p .run_venv/include

python run_sample.py --voc12_root ../../../../../datasets/Full_voc_dataset/VOC2012

#bash ./scripts/train_CIM.sh 