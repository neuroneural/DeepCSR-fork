#!/bin/bash

#SBATCH --time=02:00:00   # walltime
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks=1   # number of processes
#SBATCH --cpus-per-task=4 # number of cpus per process
#SBATCH --mem=64G  # memory

# Load libraries to run the code
SRC_DIR=/scratch1/fon022/DeepCSR/
cd ${SRC_DIR}
source /apps/miniconda3/4.3.13/etc/profile.d/conda.sh
conda deactivate
source ${SRC_DIR}/bracewell/setup.sh
# for niftyReg parallel computation
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# get job variables
ID=$SLURM_ARRAY_TASK_ID
echo "${ID} - $IN_CSV_FILE - ${OUT_DIR}"
SAMPLE_ID=$( awk -v idx="$ID" 'NR==idx {print $1; exit}' $IN_CSV_FILE )
MRI_VOL_PATH=$( awk -v idx="$ID" 'NR==idx {print $2; exit}' $IN_CSV_FILE )
LH_PIAL_PATH=$( awk -v idx="$ID" 'NR==idx {print $3; exit}' $IN_CSV_FILE )
LH_WHITE_PATH=$( awk -v idx="$ID" 'NR==idx {print $4; exit}' $IN_CSV_FILE )
RH_PIAL_PATH=$( awk -v idx="$ID" 'NR==idx {print $5; exit}' $IN_CSV_FILE )
RH_WHITE_PATH=$( awk -v idx="$ID" 'NR==idx {print $6; exit}' $IN_CSV_FILE )

echo "Running preprocessing ID=${ID}: ${SAMPLE_ID}, ${MRI_VOL_PATH}, ${LH_PIAL_PATH}, ${LH_WHITE_PATH}, ${RH_PIAL_PATH}, ${RH_WHITE_PATH}"

python preprop.py outputs.output_dir=${OUT_DIR} \
inputs.sample_id=${SAMPLE_ID} \
inputs.mri_vol_path=${MRI_VOL_PATH} \
inputs.lh_pial_path=${LH_PIAL_PATH} \
inputs.lh_white_path=${LH_WHITE_PATH} \
inputs.rh_pial_path=${RH_PIAL_PATH} \
inputs.rh_white_path=${RH_WHITE_PATH} 


# lunch command as ...
#sbatch --array=[0-MAX_JOBS]%MAX_PAR --export=OUT_DIR=<>,IN_CSV_FILE=<> <DeepCSR_ROOT>/slurm/data_preprop_job_array.q 
# example: sbatch --array=[1-2] --export=OUT_DIR=/scratch1/fon022/data_temp_npp/short/proc_data/,IN_CSV_FILE=/scratch1/fon022/data_temp_npp/short/image_surf_file.txt /scratch1/fon022/DeepCSR/slurm/data_preprop_job_array.q