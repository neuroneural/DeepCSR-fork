# load bracewell modules
conda deactivate
module load python/3.7.2
module load cuda/10.1.168
module load cudnn/v7.6.4-cuda101

# for nighres
module load jdk/1.8.0_181
export JCC_JDK=/apps/jdk/1.8.0_181/

# load python environment
source /scratch1/fon022/DeepCSR/deepcsr_venv/bin/activate

# add externals to python path
DEEPCSR_ROOT='/scratch1/fon022/DeepCSR/'
## mesh contains
export PYTHONPATH="${PYTHONPATH}:${DEEPCSR_ROOT}/external/"
export PYTHONPATH="${PYTHONPATH}:${DEEPCSR_ROOT}/external/mesh_contains/"
# nifty reg toolbox for mri registration
export NIFTYREG_PATH=${DEEPCSR_ROOT}/external/niftyreg/niftyreg-install/
export PATH=${PATH}:${NIFTYREG_PATH}/bin
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${NIFTYREG_PATH}/lib
# freesurfer v6
module load freesurfer/6.0.0

