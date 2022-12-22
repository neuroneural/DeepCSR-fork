#!/bin/bash
#SBATCH -n 1
#SBATCH -c 4 
#SBATCH --mem-per-cpu=10G
#SBATCH -p qTRDGPU
#SBATCH -t 04:00:00
#SBATCH -J prepdcsr
#SBATCH -e jobs/error%A.err
#SBATCH -o jobs/out%A.out
#SBATCH -A psy53c17
#SBATCH --mail-type=ALL
#SBATCH --mail-user=washbee1@student.gsu.edu
#SBATCH --oversubscribe

sleep 5s

#readarray -t a < dcsrarg.txt 

#echo ${a[0]}
source /usr/share/lmod/lmod/init/bash
module use /application/ubuntumodules/localmodules
module load singularity/3.10.2

singularity exec --bind /data,/data/users2/washbee/speedrun/DeepCSR-fork:/deepcsr/,/data/users2/washbee/speedrun/deepcsr-outdir/:/outdir /data/users2/washbee/containers/speedrun/deepcsr_sr.sif /deepcsr/singularity/preprop-slurm.sh $SLURM_ARRAY_TASK_ID

sleep 5s

