#!/bin/bash
. /opt/miniconda3/bin/activate deepcsr
export PYTHONPATH=/deepcsr/external/mesh_contains/ 
#source $FREESURFER_HOME/SetUpFreeSurfer.sh 
cd /deepcsr/
python train.py dataset.path=/data/users2/washbee/outdir/ \
  dataset.train_split=/deepcsr/train.txt \
  dataset.val_split=/deepcsr/val.txt \
  dataset.surfaces=['lh_pial','lh_white','rh_pial','rh_white'] \
  dataset.implicit_rpr=sdf \
  trainer.img_batch_size=15 \
  trainer.points_per_image=1024 \
  trainer.train_log_interval=10 \
  trainer.checkpoint_interval=400 \
  trainer.evaluate_interval=2000 \
  outputs.output_dir=/data/users2/washbee/deepcsr/output_dir

