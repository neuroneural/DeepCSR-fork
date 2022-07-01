#!/bin/bash
. /opt/miniconda3/bin/activate deepcsr
export PYTHONPATH=/deepcsr/external/mesh_contains/
source $FREESURFER_HOME/SetUpFreeSurfer.sh
cd /deepcsr/

#Read the string value

# Set comma as delimiter
IFS=' '

#Read the split words into an array based on space delimiter
read -a pmdata <<< "$1"

#Print the splitted words

python preprop.py outputs.output_dir=/data/users2/washbee/outdir \
    inputs.sample_id=${pmdata[0]}  \
    inputs.mri_vol_path=${pmdata[1]} \
    inputs.lh_pial_path=${pmdata[2]} \
    inputs.lh_white_path=${pmdata[3]} \
    inputs.rh_pial_path=${pmdata[4]} \
    inputs.rh_white_path=${pmdata[5]}

