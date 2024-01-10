#!/bin/bash
#SBATCH -n 1
#SBATCH -c 50
#SBATCH --mem=400g
#SBATCH -p qTRDGPU
#SBATCH -t 6999
#SBATCH -J dcsrprep
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
source activate /data/users2/washbee/anaconda3/envs/parallel
parallel singularity exec --bind /data,/data/users2/washbee/speedrun/DeepCSR-fork/:/deepcsr/ /data/users2/washbee/containers/speedrun/deepcsr_sr.sif /deepcsr/singularity/preprop.sh :::: /data/users2/washbee/speedrun/DeepCSR-fork/singularity/dcsrarg.txt &
#parallel singularity exec --bind /data,/data/users2/washbee/speedrun/DeepCSR-fork/:/deepcsr/ /data/users2/washbee/containers/speedrun/deepcsr_sr.sif echo :::: /data/users2/washbee/speedrun/DeepCSR-fork/singularity/dcsrarg.txt &
#parallel echo :::: /data/users2/washbee/speedrun/DeepCSR-fork/singularity/dcsrarg.txt &
wait

sleep 5s
