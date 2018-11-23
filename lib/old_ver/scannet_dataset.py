import pickle
import os
import numpy as np
import h5py


class ScannetDataset():
    def __init__(self, root, npoints=8192, split='train'):
        self.npoints = npoints
        self.root = root
        self.split = split

        pts = h5py.File(os.path.join(self.root , "pts_3dims.h5"))
        intensity = h5py.File(os.path.join(self.root , "intensity_3dims.h5"))
        category = h5py.File(os.path.join(self.root , "category_3dims.h5"))

        if split == 'train':
            self.scene_points_list = pts['pts'][:1000]
            self.points_intensity_list = intensity['intensity'][:1000]
            self.semantic_labels_list = category['category'][:1000]

            ###################category weight##########################
            # labelweights = np.zeros(8)
            # for seg in self.semantic_labels_list:
            #     seg = seg.reshape(-1,)
            #     tmp, _ = np.histogram(seg, range(9)) # class weight
            #     labelweights += tmp
            # labelweights = labelweights.astype(np.float32)
            # labelweights = labelweights/np.sum(labelweights)
            # self.labelweights = 1/np.log(1.2+labelweights)
            # print(self.labelweights)

        elif split == 'test':
            self.scene_points_list = pts['pts'][30000:31000]
            self.points_intensity_list = intensity['intensity'][30000:31000]
            self.semantic_labels_list = category['category'][30000:31000]

            # self.labelweights = np.ones(8)

    def __getitem__(self, index):
        # print("index : " ,index)
        point_set = self.scene_points_list[index] #point cloud
        semantic_seg = self.semantic_labels_list[index].astype(np.int32).reshape(-1,) #reshape here #label
        intensity = self.points_intensity_list[index]

        print(point_set.shape)
        print(semantic_seg.shape)

        coordmax = np.max(point_set,axis=0) #find the max coord in whole list
        coordmin = np.min(point_set,axis=0)

        smpmin = np.maximum(coordmax-[1.5, 1.5, 3.0], coordmin)

        smpmin[2] = coordmin[2]
        smpsz = np.minimum(coordmax-smpmin, [1.5, 1.5, 3.0])
        smpsz[2] = coordmax[2]-coordmin[2]
        cur_semantic_seg = None
        cur_point_set = None
        cur_intensity = None

        for i in range(10):
            curcenter = point_set[np.random.choice(len(semantic_seg),1)[0],:]
            curmin = curcenter-[0.75, 0.75, 1.5]
            curmax = curcenter+[0.75, 0.75, 1.5]
            curmin[2] = coordmin[2]
            curmax[2] = coordmax[2]
            curchoice = np.sum((point_set >= (curmin-0.2))*(point_set <= (curmax+0.2)),axis=1) == 3
            cur_point_set = point_set[curchoice, :]
            cur_semantic_seg = semantic_seg[curchoice]
            cur_intensity = intensity[curchoice, :]
            if len(cur_semantic_seg) == 0:
                continue
            mask = np.sum((cur_point_set >= (curmin-0.01))*(cur_point_set <= (curmax+0.01)), axis=1) == 3
            vidx = np.ceil((cur_point_set[mask, :]-curmin)/(curmax-curmin)*[31.0, 31.0, 62.0])
            vidx = np.unique(vidx[:, 0] * 31.0 * 62.0 + vidx[:, 1] * 62.0 + vidx[:, 2])
            isvalid = np.sum(cur_semantic_seg > 0)/len(cur_semantic_seg) >= 0.7 and len(vidx)/31.0/31.0/62.0 >= 0.02
            if isvalid:
                break
        choice = np.random.choice(len(cur_semantic_seg), self.npoints, replace=True) #(256,)
        print(choice.shape)
        point_set = cur_point_set[choice,:]
        semantic_seg = cur_semantic_seg[choice]
        intensity = cur_intensity[choice, :]
        
        return point_set, semantic_seg, intensity

    def __len__(self):
        return len(self.scene_points_list)
