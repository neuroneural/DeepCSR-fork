#!/bin/bash
. /opt/miniconda3/bin/activate deepcsr
export PYTHONPATH=/deepcsr/external/mesh_contains/ 
#source $FREESURFER_HOME/SetUpFreeSurfer.sh 
cd /deepcsr/
readarray -t a < /deepcsr/singularity/test_ids.csv
echo 'array is read'
python predict.py inputs.mri_id=${a[${SLURM_ARRAY_TASK_ID}]} \
inputs.mri_vol_path=/data/users2/washbee/speedrun/deepcsr-preprocessed/${a[${SLURM_ARRAY_TASK_ID}]}/mri.nii.gz \
inputs.model_checkpoint=/data/users2/washbee/speedrun/outputdirs/deepcsr-output_dir/best_model.pth \
inputs.model_surfaces=['lh_pial','lh_white','rh_pial','rh_white'] \
generator.resolution=256 \
outputs.output_dir=/data/users2/washbee/speedrun/outputdirs/deepcsr-output_dir-timing/checkpoints/test-set/ \
outputs.save_all=False

