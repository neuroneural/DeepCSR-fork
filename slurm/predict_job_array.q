#!/bin/bash

#SBATCH --time=2:00:00 # walltime
#SBATCH --nodes=1 # Number of computer nodes
#SBATCH --ntasks-per-node=4 # number of process per node
#SBATCH --cpus-per-task=2 # number of threads per process
#SBATCH --mem-per-cpu=32G # memory per node
#SBATCH --gres=gpu:1 # number of gpus


# Load libraries to run the code
SRC_DIR=/scratch1/fon022/DeepCSR/
cd ${SRC_DIR}
source /apps/miniconda3/4.3.13/etc/profile.d/conda.sh
conda deactivate
source ${SRC_DIR}/bracewell/setup.sh

# get job variables
ID=$SLURM_ARRAY_TASK_ID
echo "${ID} - ${CONFIG} - $IN_FILE - ${OUT_DIR}"
MRI_ID=$( awk -v idx="$ID" 'NR==idx {print $1; exit}' $IN_FILE )
MRI_VOL_PATH=$( awk -v idx="$ID" 'NR==idx {print $2; exit}' $IN_FILE )


echo "Running Prediction ID=${ID}: ${MRI_ID}, ${MRI_VOL_PATH}"

python predict.py  user_config=${CONFIG} inputs.mri_id=${MRI_ID} inputs.mri_vol_path=${MRI_VOL_PATH} outputs.output_dir=${OUT_DIR}/${MRI_ID}/ 

#launch command as ...
#sbatch --array=[0-MAX_JOBS]%MAX_PAR --export=CONFIG=<>,OUT_DIR=<>,IN_FILE=<> <DeepCSR_ROOT>/slurm/predict_job_array.q 
