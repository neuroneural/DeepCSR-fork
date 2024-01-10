#!/bin/bash
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=40g
#SBATCH -p qTRDGPUH
#SBATCH --gres=gpu:V100:1
#SBATCH -t 4-00:00
#SBATCH -J deepcsr
#SBATCH -e jobs/error%A.err
#SBATCH -o jobs/out%A.out
#SBATCH -A psy53c17
#SBATCH --mail-type=ALL
#SBATCH --mail-user=washbee1@student.gsu.edu
#SBATCH --oversubscribe

sleep 5s

source /usr/share/lmod/lmod/init/bash
module use /application/ubuntumodules/localmodules
module load singularity/3.10.2
singularity exec --nv --bind /data,/data/users2/washbee/speedrun/DeepCSR-fork/:/deepcsr/,/data/users2/washbee/speedrun/deepcsr-preprocessed:/subj /data/users2/washbee/containers/speedrun/deepcsr_sr.sif /deepcsr/singularity/train.sh &

wait

sleep 10s
