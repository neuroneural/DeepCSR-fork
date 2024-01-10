#!/bin/bash
. /opt/miniconda3/bin/activate deepcsr
export PYTHONPATH=/deepcsr/external/mesh_contains/
source $FREESURFER_HOME/SetUpFreeSurfer.sh
cd /deepcsr/

#Read the string value

# Set comma as delimiter
IFS=' '

#Read the split words into an array based on space delimiter
echo 1 is $1
read -a pmdata <<< "$1"

#Print the splitted words
echo pmdata is ${pmdata[0]} ${pmdata[1]} ${pmdata[2]} ${pmdata[3]} ${pmdata[4]} ${pmdata[5]}

python preprop.py outputs.output_dir=/data/users2/washbee/speedrun/deepcsr-prep-demo \
    inputs.sample_id=${pmdata[0]}  \
    inputs.mri_vol_path=${pmdata[1]} \
    inputs.lh_pial_path=${pmdata[2]} \
    inputs.lh_white_path=${pmdata[3]} \
    inputs.rh_pial_path=${pmdata[4]} \
    inputs.rh_white_path=${pmdata[5]}

