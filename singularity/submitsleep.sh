#!/bin/bash
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=1g
#SBATCH -p qTRDGPUH
#SBATCH -t 0-01:00
#SBATCH -J tasktest
#SBATCH -e error%A-%a.err
#SBATCH -o out%A-%a.out
#SBATCH -A psy53c17
#SBATCH --mail-type=ALL
#SBATCH --mail-user=washbee1@student.gsu.edu
#SBATCH --oversubscribe

sleep 5s

echo $SLURM_ARRAY_TASK_ID 

sleep 5s
