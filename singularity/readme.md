# How to run the preprocessing
      parallel singularity exec --bind /data:/data/,/home:/home/,/home/users/washbee1/projects/deepcsr:/deepcsr/ /data/users2/washbee/containers/deepcsr.sif /deepcsr/singularity/preprop.sh :::: dcsrarg.txt

# Time and resources 
Takes about 15 minutes to run 1 patient (and patients will be run in parallel), and is cpu heavy instead of ram heavy. You will not see output for at least 14 minutes, but can monitor with top.
# NO GPU REQUIRED
You won't need gpus for preprocessing. 
# SLURM SCRIPT ADDED 
            SBATCH submit.sh

I think this tends to fail if you give it more than 39 cpus on the trends cluster setup. 
