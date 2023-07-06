#!/bin/bash
. /opt/miniconda3/bin/activate deepcsr
export PYTHONPATH=/deepcsr/external/mesh_contains/ 
#source $FREESURFER_HOME/SetUpFreeSurfer.sh 
cd /deepcsr/
readarray -t a < /deepcsr/singularity/benchmarks/bm.csv
echo 'array is read'
python bm.predict.py inputs.mri_id=${a[0]} \
inputs.mri_vol_path=/data/users2/washbee/speedrun/deepcsr-preprocessed/${a[0]}/mri.nii.gz \
inputs.model_checkpoint=/data/users2/washbee/speedrun/outputdirs/deepcsr-output_dir/best_model.pth \
inputs.model_surfaces=['lh_pial','lh_white','rh_pial','rh_white'] \
generator.resolution=256 \
outputs.output_dir=/data/users2/washbee/speedrun/outputdirs/deepcsr-output_dir-timing/checkpoints/bmtest-set/ \
outputs.save_all=False

