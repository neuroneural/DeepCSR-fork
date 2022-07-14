#!/bin/bash
#SBATCH -n 1
#SBATCH -c 15
#SBATCH --mem=40g
#SBATCH -p qTRDGPUH
#SBATCH --gres=gpu:v100:1
#SBATCH --nodelist=trendsdgx003.rs.gsu.edu 
#SBATCH -t 3-00:00
#SBATCH -J deepcsr
#SBATCH -e /data/users2/washbee/deepcsr/jobs/error%A.err
#SBATCH -o /data/users2/washbee/deepcsr/jobs/out%A.out
#SBATCH -A PSYC0002
#SBATCH --mail-type=ALL
#SBATCH --mail-user=washbee1@student.gsu.edu
#SBATCH --oversubscribe

sleep 5s

singularity exec --nv --bind /data:/data/,/home:/home/,/home/users/washbee1/projects/deepcsr:/deepcsr/,/data/users2/washbee/outdir:/subj /data/users2/washbee/containers/deepcsr.sif /deepcsr/singularity/train.sh &

wait

sleep 10s
