"""
#make the training float precistion equal to the testing

"""

from stat import S_ISREG, ST_CTIME, ST_MODE
import numpy as np
import os, sys, time
import pandas as pd
import h5py
import gc; gc.enable()
from tqdm import tqdm

mode = "TrainSet"
save_h5_name = "train_r5"
current_path = os.getcwd()
file_path = os.path.join(current_path, '{}/all_data/'.format(mode))
eda_file = os.path.join(current_path, os.path.join("./EDA", "eda_r50.txt")) # os.path.join("./EDA", "eda_r50.txt")

class HDF5Store(object):

    def __init__(self, datapath, pts,intensity,category, dtype='f16', compression="gzip", chunk_len=1):
        self.datapath = datapath
        self.dataset_xyz = pts
        self.dataset_i = intensity
        self.dataset_c = category
        self.i = 1
        with h5py.File(self.datapath, mode='w', libver='latest') as h5f:

            self.gset_xyz = h5f.create_group("base")

            self.dset_xyz = self.gset_xyz.create_dataset(
                "pts",
                self.dataset_xyz.shape,
                maxshape=(None, None, 3),
                dtype=dtype,
                compression=compression,
                chunks=self.dataset_xyz.shape)

            self.dset_i = self.gset_xyz.create_dataset(
                "intensity",
                self.dataset_i.shape,
                maxshape=(None, None),
                dtype=dtype,
                compression=compression,
                chunks=self.dataset_i.shape)

            self.dset_c = self.gset_xyz.create_dataset(
                "category",
                self.dataset_c.shape,
                maxshape=(None, None),
                dtype=dtype,
                compression=compression,
                chunks=self.dataset_c.shape)


            self.dset_xyz[:,:,:] = self.dataset_xyz
            print(self.dset_xyz.shape)

            self.dset_i[:,:] = self.dataset_i
            print(self.dset_i.shape)

            self.dset_c[:,:] = self.dataset_c
            print(self.dset_c.shape)

    def append(self, values_xyz, values_i, values_c):
        with h5py.File(self.datapath, mode='a', libver='latest') as h5f:
            dset_xyz = h5f['base/pts']
            dset_i = h5f['base/intensity']
            dset_c = h5f['base/category']
            # print("origin : ", dset_xyz.shape)
            # print("adding : ", dset_xyz.shape)
            dset_xyz.resize(dset_xyz.shape[0] + values_xyz.shape[0], axis = 0)
            dset_i.resize(dset_i.shape[0] + values_i.shape[0], axis = 0)
            dset_c.resize(dset_c.shape[0] + values_c.shape[0], axis = 0)
            # print("resized to : ", dset_xyz.shape)
            dset_xyz[-values_xyz.shape[0]:,:,:] = values_xyz
            dset_i[-values_i.shape[0]:,:] = values_i
            dset_c[-values_c.shape[0]:,:] = values_c

            # print(dset)
            # print("\n")
            self.i += 1
            if self.i % 100 == 0:
            	print("**************")
            	print(dset_xyz)
            	print("**************")

            h5f.flush()
            gc.collect()


#read csv path
with open(eda_file) as f:
    txt = f.readlines()
    print("Runing with {} files".format(len(txt)))

for i in tqdm(range(len(txt))):

    try:
        df = pd.read_csv(os.path.join(file_path, txt[i][:-1]), float_precision = 'round_trip') #make the training float precistion equal to the testing
    except Exception as e:
        print("NO DIR FOUND")
        raise
    pts = np.expand_dims(np.array(df[['x','y','z']].values)[:52000,:], 0) # fixe 5w points
    intensity = np.expand_dims(np.array(df['i'].values)[:52000], 0) # fixe 5w points
    category = np.expand_dims(np.array(df['c'].values)[:52000], 0) # fixe 5w points

    if i == 0:
        hdf5_store = HDF5Store('{}/{}.h5'.format(mode, save_h5_name), pts, intensity, category)

    else:
        shape = hdf5_store.append(pts, intensity, category)
