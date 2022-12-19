#!/bin/bash
. /opt/miniconda3/bin/activate deepcsr
export PYTHONPATH=/deepcsr/external/mesh_contains/ 
#source $FREESURFER_HOME/SetUpFreeSurfer.sh 
cd /deepcsr/
python predict.py inputs.mri_id=201818 \
inputs.mri_vol_path=/data/users2/washbee/outdir/201818/mri.nii.gz \
inputs.model_checkpoint=/data/users2/washbee/deepcsr/output_dir/best_model.pth \
inputs.model_surfaces=['lh_pial','lh_white','rh_pial','rh_white'] \
generator.resolution=64 \
outputs.output_dir=/data/users2/washbee/deepcsr/output_dir/checkpoints/predict_debug/ \
outputs.save_all=False


