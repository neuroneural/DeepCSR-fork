#!/bin/bash
#SBATCH -n 1
#SBATCH -c 39
#SBATCH --mem=100g
#SBATCH -p qTRDHM
#SBATCH --nodelist=trendsmn003.rs.gsu.edu
#SBATCH -t 6999
#SBATCH -J dcsrprep
#SBATCH -e jobs/error%A.err
#SBATCH -o jobs/out%A.out
#SBATCH -A PSYC0002
#SBATCH --mail-type=ALL
#SBATCH --mail-user=washbee1@student.gsu.edu
#SBATCH --oversubscribe

sleep 5s

parallel singularity exec --bind /data:/data/,/home:/home/,/home/users/washbee1/projects/deepcsr:/deepcsr/ /data/users2/washbee/containers/deepcsr.sif /deepcsr/singularity/preprop.sh :::: /home/users/washbee1/projects/deepcsr/singularity/dcsrarg.txt
wait

sleep 10s
