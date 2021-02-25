

# DeepCSR
![Sample 3D Outer Surface](resources/sample.gif)

This repository contains the official implementation of the paper ***"DeepCSR: A 3D Deep Learning Approach for Cortical Surface Reconstruction"***. It is published in the WACV21 conference where it received **the best paper award in the algorithm track**. Links: [Paper](https://openaccess.thecvf.com/content/WACV2021/papers/Santa_Cruz_DeepCSR_A_3D_Deep_Learning_Approach_for_Cortical_Surface_Reconstruction_WACV_2021_paper.pdf), [Talk](https://youtu.be/IAkadGjkqbQ), and [Slides](https://rfsantacruz.com/files/pdfs/wacv21_deepcsr_slides.pdf).

If you find our code or paper useful, please cite
```
@InProceedings{SantaCruz:WACV21:DeepCSR,
    author    = {Santa Cruz, Rodrigo and Lebrat, Leo and Bourgeat, Pierrick and Fookes, Clinton and Fripp, Jurgen and Salvado, Olivier},
    title     = {DeepCSR: A 3D Deep Learning Approach for Cortical Surface Reconstruction},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2021},
    pages     = {806-815}
}
```

For further information or questions, please email rodrigo.santacruz@csiro.au.
See below detailed usage instructions:


## Installation

This software was developed using a High Computing Platform with SUSE Linux Enterprise 12, Nvidia P100 GPUs, python 3.7.2, CUDA 10.1.168, CUDNN v7.6.4, and Virtualenv to manage python dependencies. The installation is as follows,

1. Create python environment:
```
virtualenv --python=python3.7 ./deepcsr_venv/
source ./deepcsr_venv/bin/activate
```

2. Install python libraries using PIP (you can also check the requirement files generated in ./requirements.txt):
```
pip install torch==1.7.0+cu101 torchvision==0.8.1+cu101 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install trimesh 
pip install hydra-core --upgrade
pip install Cython
pip install scipy
pip install nibabel
pip install tensorboard
pip install scikit-image
pip install joblib
pip install rtree
```

3. Some external libraries need to be installed from source in ./external folder. 
    1. Mesh contains script from [OccupancyNetworks](https://github.com/autonomousvision/occupancy_networks) and also available at [https://gist.github.com/LMescheder/b5e03ffd1bf8a0dfbb984cacc8c99532](https://gist.github.com/LMescheder/b5e03ffd1bf8a0dfbb984cacc8c99532). 
    ```
    cd external/mesh_contains/
    python setup.py build_ext --inplace
    export PYTHONPATH="${PYTHONPATH}:${DEEPCSR_ROOT}/external/mesh_contains/"
    ```
    
    2. **NiftyReg v1.5.58** medical image registration toolbox. Please follow the instructions in [http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftyReg_install](http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftyReg_install).
    
    3. **FreeSurfer V6** (for data preprocessing), please follow instructions in [https://surfer.nmr.mgh.harvard.edu/fswiki/rel6downloads).](https://surfer.nmr.mgh.harvard.edu/fswiki/rel6downloads).
    
    4. **NigHRes V1.1** (for topology correction), please follow instruction in [https://nighres.readthedocs.io/en/latest/installation.html](https://nighres.readthedocs.io/en/latest/installation.html).


## Usage
This source code uses [Hydra configuration files](https://hydra.cc/docs/intro/) and [Slurm workload manager](https://slurm.schedmd.com/documentation.html) to run jobs in a High-performance computing (HPC) platform. Below, I will describe the main scripts, their configuration files, and slurm script used to run experiments in our HPC. You may need to perform small adaptations to them in order to run experiments in your computing platform.

In summary, the main scripts are in the root directory, the default configuration files for these scripts are in ./configs/, and slurm job scripts are in ./slurm folder. **Our default configuration files are densely commented explaining in details each one of the scripts parameters.**


### Preprocessing Data
Before training a DeepCSR model, one need to compute implicit surface representations in a brain template space from a dataset of MRI and aligned cortical surface pairs. It can be done for a given subject as follows,
```
python preprop.py outputs.output_dir=<OUT_DIR> \
    inputs.sample_id=<SAMPLE_ID> \
    inputs.mri_vol_path=<MRI_VOL_PATH> \
    inputs.lh_pial_path=<LH_PIAL_PATH> \
    inputs.lh_white_path=<LH_WHITE_PATH> \
    inputs.rh_pial_path=<RH_PIAL_PATH> \
    inputs.rh_white_path=<RH_WHITE_PATH> 
```
where the paths to the MRI, surfaces meshes, output directory and sample id are passed by command line. More options (e.g., the brain template used) can be found at *./configs/preprop.yaml*. In order to run this script for a batch of subjects to create a dataset, we used slurm job array as described in *./slurm/data_preprop_job_array.q*. It basically consumes a `\t` separated file where each line represent one subject and the columns are its respective id, MRI path, left hemisphere pial (outer) surface path, left hemis. white (inner) surface path, right hemis. pial surface path, and right hemis. white surface path.


### Training Model
Once the data is preprocessed, we can train DeepCSR from scratch as follows,
```
python train.py outputs.output_dir=<OUT_DIR> user_config=<CONFIG>
```
where the output directory and configuration file is passed by command line. The passed configuration file is used to override the default training options in *./configs/train.yaml* (e.g., batch size or implicit representation used). Please check this configuration file for more information. We also provide the slurm script *./slurm/train_job.q* to launch the training.


### Generating Surfaces
For generating surfaces for a given input MRI using a trained DeepCSR model, execute the following,
```
python predict.py  user_config=<CONFIG> inputs.mri_id=<MRI_ID> inputs.mri_vol_path=<MRI_VOL_PATH> outputs.output_dir=<OUT_DIR> 
```
where a configuration file, MRI id, MRI path and output directory is passed by command line. As before, check the configuration file in *./configs/predict.yaml* to extra options. The generated surfaces will be saved in the passed output directory using the passed mri_id as a filename prefix.
In order to scale up this script for a batch of MRIs, we use the slurm job array script in *./slurm/predict_job_array.q*. Again, it consumes a `\t` separated file where each lines is one subject and the columns are its respective id and MRI path.


### Evaluating Generated Surfaces
We also provide a simple surface evaluation script to check the performance of the trained model. The evaluation procedure can be called as follows,
```
python eval.py user_config=<CONFIG> outputs.output_dir=<OUT_DIR>
```
and its default configuration file is *./configs/eval.yaml*, and associated slurm job script is *./slurm/eval_job_array.q*.







