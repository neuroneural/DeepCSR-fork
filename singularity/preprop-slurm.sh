#!/bin/bash
. /opt/miniconda3/bin/activate deepcsr
export PYTHONPATH=/deepcsr/external/mesh_contains/
source $FREESURFER_HOME/SetUpFreeSurfer.sh
cd /deepcsr/

#Read the string value

# Set comma as delimiter
IFS=' '

readarray -t a < /data/users2/washbee/speedrun/DeepCSR-fork/singularity/dcsrarg.txt
#echo ${a[0]}
read -a pmdata <<< "${a[$1]}"
echo "pmdata" $pmdata ${pmdata[0]} ${pmdata[1]} 

#Print the splitted words


python preprop.py outputs.output_dir=/data/users2/washbee/speedrun/deepcsr-preprocessed \
    inputs.sample_id=${pmdata[0]}  \
    inputs.mri_vol_path=${pmdata[1]} \
    inputs.lh_pial_path=${pmdata[2]} \
    inputs.lh_white_path=${pmdata[3]} \
    inputs.rh_pial_path=${pmdata[4]} \
    inputs.rh_white_path=${pmdata[5]}

