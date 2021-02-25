
# load bracewell modules
conda deactivate
module load python/3.7.2
module load cuda/10.1.168
module load cudnn/v7.6.4-cuda101

# create environment and redirect cache to local folder
virtualenv --python=python3.7 ./deepcsr_venv/
source ./deepcsr_venv/bin/activate
mkdir ./pip_cache/
export PIP_CACHE_DIR=/scratch1/fon022/DeepCSR/pip_cache/


# install python libs
pip install torch==1.7.0+cu101 torchvision==0.8.1+cu101 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install trimesh 
pip install hydra-core --upgrade
pip install Cython
pip install scipy
pip install nibabel
pip install tensorboard
pip install scikit-image
pip install joblib
pip install rtree

## install externals
# mesh contains
cd external/mesh_contains/
python setup.py build_ext --inplace

# install nifty reg as described in http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftyReg_install

# install nighres as described in https://nighres.readthedocs.io/en/latest/installation.html and run
module load jdk/1.8.0_181
export JCC_JDK=/apps/jdk/1.8.0_181/

# install freesurfer v6
module load freesurfer/6.0.0