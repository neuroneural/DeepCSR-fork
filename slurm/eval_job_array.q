#!/bin/bash

#SBATCH --time=12:00:00 # walltime
#SBATCH --nodes=1 # Number of computer nodes
#SBATCH --ntasks-per-node=16 # number of process per node
#SBATCH --cpus-per-task=1 # number of threads per process
#SBATCH --mem-per-cpu=16G # memory per node


# Load libraries to run the code
SRC_DIR=/scratch1/fon022/DeepCSR/
cd ${SRC_DIR}
source /apps/miniconda3/4.3.13/etc/profile.d/conda.sh
conda deactivate
source ${SRC_DIR}/bracewell/setup.sh

python eval.py user_config=${CONFIG} outputs.output_dir=${OUT_DIR} 

# launch command as ...
#sbatch --export=CONFIG=<>,OUT_DIR=<> <DeepCSR_ROOT>/slurm/eval_job_array.q 
