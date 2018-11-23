import h5py

mode = 'train_all_data_beta1'
key = 'pts'
def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    print(f[key].shape[0])
    # data = f[key][:]
    return #data

def load_group_h5(h5_filename):
    f = h5py.File(h5_filename)
    print(f['base/pts'])
    print(f['base/intensity'])
    print(f['base/category'])

    return #data

# data = load_h5('./TrainSet/{}.h5'.format(mode))
load_group_h5('./TrainSet/{}.h5'.format(mode))
# for i in range(88):
# 	print(data[i])

# print(data.shape)