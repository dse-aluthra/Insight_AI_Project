# Scalable Training 3D Medical-Imaging
Develop techniques for Scalable Training for 3D Medical Imaging, entailing Distributing-Computing approach by doing Data & Model Parallelization to expedite training process.
- https://platform.insightdata.com/projects/scalable-training-3d-medical-imaging


## Project Folder Structure:
- **./src** : /model;
              /preprocess;
              /notebooks
- **./configs** : /model.ini;
                  /crop_nodules_3d.ini
- **data** :  /raw;
              /preprocessed
- **build** : /create_conda_env.sh;
              /environment.yml

## Setup
Clone repository and update python path
```
repo_name=https://github.com/dse-aluthra/Insight_AI_Project
username=dse-aluthra
#Clone master
git clone https://github.com/$username/$repo_name
cd $repo_name

#Clone branch
branch_id=model-parallelize
git clone -b $branch_id  https://github.com/$username/$repo_name

```
Create new development branch and switch onto it
```
branch_name=dev-readme_requisites-20180917
# Name branch, of the form 'dev-feature_name-date_of_creation'}}
git checkout -b $branch_name
```

## Requisites
- Python 3.x
- Tensorflow 1.10
- Keras
- Numpy
- SITK (pip install sitk)
- gpustat (pip install gpustat)
- dstat (pip install dstat)
- psutils (pip install psutils)
- HDF5 i.e h5py (pip install h5py)
- Resnet3d  (pip install git+https://github.com/JihongJu/keras-resnet3d.git)

## Build Environment
 - Run ./build/create_conda_env.sh (which read environment.yml) to create the conda environment for preprocessing raw data

## Configs
- ./configs/crop_nodules_3d.ini
- - Configuration files to specify folder path to raw data, supporting configuration files and destination, else default values aligned to existing folder structure from git repo will be used.

- ./configs/model.ini
- - Configuration files used by the model-training scripts
- -cParameters like BATCH_SIZE, GPUID, path to HDF5 file can be specify (else default values are considered)

## Executing the scripts
- > python ./src/preprocess/crop_nodules_3d.py
- This script will preprocess the raw data and create HDF5 file.

- > python ./src/model/modelTrng_No_GPU.py
- This will initiate the training process for the model with no GPU usage


- This will initiate the training process for the model
      python ./src/model/modelTrng_with_GPU.py
