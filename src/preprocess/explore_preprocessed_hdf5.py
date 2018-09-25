# !/usr/bin/python
import os
import numpy as np
import pandas as pd
from ipywidgets import interact
from configparser import ConfigParser
import h5py # Read the Docs: http://docs.h5py.org/en/latest/index.html

config = ConfigParser()
config.read('../../configs/crop_nodules_3d.ini')

PREPROCESSED_PATH = config.get('remote', 'PREPROCESSED_PATH')

hdf5_file_filename = '32x32x32-patch.hdf5'
path_to_hdf5 = PREPROCESSED_PATH + hdf5_file_filename
hdf5_file = h5py.File(path_to_hdf5, 'r') # open in read-only mode

print("Valid hdf5 file in 'read' mode: " + str(hdf5_file))
file_size = os.path.getsize(path_to_hdf5)
print('Size of hdf5 file: {:.3f} GB'.format(file_size/2.0**30))

print('Dataset info and some real data:')
for name in [key for key in hdf5_file.keys()]:
    print(name)
    print(hdf5_file[name])
    # print(hdf5_file[name][0:2])

idx_class0 = np.where(hdf5_file["output"][:,0] == 0)[0]
idx_class1 = np.where(hdf5_file["output"][:,0] == 1)[0]

print('Class 0 count **: {}'.format( len(idx_class0) ) )
print('Class 1 count ** : {}'.format( len(idx_class1) ) )
