from __future__ import division
from __future__ import print_function

import os
import numpy as np
import time
import tensorflow as tf
import argparse
import h5py
from tqdm import tqdm
import pandas as pd

import tf_utils.provider as provider
import models.pointSIFT_pointnet as SEG_MODEL

parser = argparse.ArgumentParser()
parser.add_argument('--max_epoch', type=int, default=1000, help='epoch to run[default: 1000]')
parser.add_argument('--batch_size', type=int, default=16, help='batch size during training[default: 32')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='initial learning rate[default: 1e-3]')
parser.add_argument('--save_path', default='/data/lecui/3d_drive/Models', help='model param path')
parser.add_argument('--data_path', default='/data/lecui/3d_drive/TrainSet', help='scannet dataset path')
parser.add_argument('--train_log_path', default='log/pointSIFT_train')
parser.add_argument('--test_log_path', default='log/pointSIFT_test')
parser.add_argument('--gpu_num', type=int, default=2, help='number of GPU to train')

# basic params..

FLAGS = parser.parse_args()
BATCH_SZ = FLAGS.batch_size
LEARNING_RATE = FLAGS.learning_rate
MAX_EPOCH = FLAGS.max_epoch
SAVE_PATH = FLAGS.save_path
DATA_PATH = FLAGS.data_path
TRAIN_LOG_PATH = FLAGS.train_log_path
TEST_LOG_PATH = FLAGS.test_log_path
GPU_NUM = FLAGS.gpu_num
BATCH_PER_GPU = BATCH_SZ // GPU_NUM

NUM_CLASS = 8
TRAIN_FILE = 'train_r10.h5' #'train_r10_subsamp_1w5.h5'

# lr params..
DECAY_STEP = 200000
DECAY_RATE = 0.7

# bn params..
BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

class SegTrainer(object):
    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.train_sz = 11000
        self.test_sz = 2000
        self.point_sz = 8192 #8192 #5120 #40000

        # Jim Added: total sample weight of all class
        self.labelweights = None

        # batch loader init....
        self.batch_loader = None
        self.batch_sz = BATCH_SZ

        # net param...
        self.point_pl = None
        self.label_pl = None
        self.smpws_pl = None
        self.intensity = None
        self.is_train_pl = None
        self.ave_tp_pl = None
        self.net = None
        self.end_point = None
        self.bn_decay = None
        self.loss = None
        self.optimizer = None
        self.train_op = None
        self.predict = None
        self.TP = None
        self.batch = None  # record the training step..

        # summary
        self.ave_tp_summary = None

        # list for multi gpu tower..
        self.tower_grads = []
        self.net_gpu = []
        self.total_loss_gpu_list = []

    def get_learning_rate(self):
        learning_rate = tf.train.exponential_decay(LEARNING_RATE,
                                                   self.batch * BATCH_SZ,
                                                   DECAY_STEP,
                                                   DECAY_RATE,
                                                   staircase=True)
        learning_rate = tf.maximum(learning_rate, 1e-5)
        tf.summary.scalar('learning rate', learning_rate)
        return learning_rate

    def get_bn_decay(self):
        bn_momentum = tf.train.exponential_decay(BN_INIT_DECAY,
                                                 self.batch * BATCH_SZ,
                                                 BN_DECAY_DECAY_STEP,
                                                 BN_DECAY_DECAY_RATE,
                                                 staircase=True)
        bn_momentum = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
        tf.summary.scalar('bn_decay', bn_momentum)
        return bn_momentum

    def get_batch_wdp(self, pts, category, idxs, start_idx, end_idx):

        bsize = end_idx - start_idx
        batch_data = np.zeros((bsize, self.point_sz, 3))
        batch_label = np.zeros((bsize, self.point_sz), dtype=np.int32)
        batch_smpw = np.zeros((bsize, self.point_sz), dtype=np.float32)

        for i in range(bsize):

            _pts = pts[idxs[i + start_idx]]
            _category = category[idxs[i + start_idx]]

            batch_data[i, ...] = _pts
            batch_label[i, :] = _category
            for class_num in range(8):
                batch_smpw[batch_label == class_num] = self.labelweights[class_num]

            dropout_ratio = np.random.random() * 0.875  # 0-0.875
            drop_idx = np.where(np.random.random((ps.shape[0])) <= dropout_ratio)[0]

            batch_data[i, drop_idx, :] = batch_data[i, 0, :]
            batch_label[i, drop_idx] = batch_label[i, 0]
            batch_smpw[i, drop_idx] *= 0

        return batch_data, batch_label, batch_smpw

    def get_batch(self, pts, category, idxs, start_idx, end_idx):

        bsize = end_idx - start_idx
        batch_data = np.zeros((bsize, self.point_sz, 3))
        batch_label = np.zeros((bsize, self.point_sz), dtype=np.int32)
        batch_smpw = np.zeros((bsize, self.point_sz), dtype=np.float32)

        for i in range(bsize):

            _pts = pts[idxs[i + start_idx]]
            _category = category[idxs[i + start_idx]]

            _pts, _category, _smpw = self.t_net(_pts, _category) #t-net

            batch_data[i, ...] = _pts
            batch_label[i, :] = _category
            batch_smpw[i, :] = _smpw

            # for class_num in range(8):
                # batch_smpw[batch_label == class_num] = self.labelweights[class_num]

        return batch_data, batch_label, batch_smpw

    def get_evl_batch(self, pts, category, idxs, start_idx, end_idx):

        bsize = end_idx - start_idx
        batch_data = np.zeros((bsize, 52000, 3))
        batch_label = np.zeros((bsize, 52000), dtype=np.int32)
        batch_smpw = np.zeros((bsize, 52000), dtype=np.float32)

        for i in range(bsize):

            _pts = pts[idxs[i + start_idx]]
            _category = category[idxs[i + start_idx]]

            batch_data[i, ...] = _pts
            batch_label[i, :] = _category

            for class_num in range(8):
                batch_smpw[batch_label == class_num] = self.labelweights[class_num]

        return batch_data, batch_label, batch_smpw     

    def t_net(self, point_set, semantic_seg):

        coordmax = np.max(point_set,axis=0) #find the max coord in whole list
        coordmin = np.min(point_set,axis=0)

        #change the range of coordmax

        # smpmin = np.maximum(coordmax-[5, 5, 2], coordmin)

        # smpmin[2] = coordmin[2]
        # smpsz = np.minimum(coordmax-smpmin, [5, 5, 2])
        # smpsz[2] = coordmax[2]-coordmin[2]
        cur_semantic_seg = None
        cur_point_set = None

        for i in range(90):

            curcenter = point_set[np.random.choice(len(semantic_seg),1)[0],:] #random choice one point in the cube (get point index)
            # if np.any(curcenter > 80) or np.any(curcenter < -80) or curcenter[2] <= -30 or curcenter[2] >= 30:
                # print("curcenter : ", curcenter)

            # range condition (make suer the random point restrict in a certain range)
            while np.any(curcenter > 80) or np.any(curcenter < -80) or curcenter[2] <= -30 or curcenter[2] >= 30:
                curcenter = point_set[np.random.choice(len(semantic_seg),1)[0],:]
                # print("range_condition : ", curcenter)

            curmin = curcenter-[2.5, 2.5, 1]
            curmax = curcenter+[2.5, 2.5, 1]
            curmin[2] = coordmin[2]
            curmax[2] = coordmax[2]
            curchoice = np.sum((point_set >= (curmin-1))*(point_set <= (curmax+1)),axis=1) == 3 #find the point in the cube (wider : 5)
            cur_point_set = point_set[curchoice, :]
            cur_semantic_seg = semantic_seg[curchoice]
            if len(cur_semantic_seg) == 0:
                continue
            mask = np.sum((cur_point_set >= (curmin-0.1))*(cur_point_set <= (curmax+0.1)), axis=1) == 3 #find the point in the cube (wider : 1)
            # vidx = np.ceil((cur_point_set[mask, :]-curmin)/(curmax-curmin)*[1.0, 1.0, 1.0]) #point cloud to voxel
            # vidx = np.unique(vidx[:, 0] * 1.0 * 1.0 + vidx[:, 1] * 1.0 + vidx[:, 2])
            # print("vid", len(vidx))
            # cube has >= 70% valid label
            isvalid = np.sum(cur_semantic_seg > 0)/len(cur_semantic_seg) >= 0.5 #and len(vidx)/1.0/1.0/1.0 >= 0.02 
            if isvalid:
                break
            if i == 89 and np.sum(cur_semantic_seg > 0) == 0 :
                print("did not find valid points")

        choice = np.random.choice(len(cur_semantic_seg), self.point_sz, replace=True)
        point_set = cur_point_set[choice,:]

        semantic_seg = cur_semantic_seg[choice]
        mask = mask[choice]
               
        sample_weight = self.labelweights[semantic_seg]
        # print("sample_weight_1 : ", sample_weight)
        # print("sample_weight_1 : ", sample_weight.shape)
        # sample_weight *= mask
        # print("sample_weight_2 : ", sample_weight)
        # print("sample_weight_2 : ", sample_weight.shape)

        return point_set, semantic_seg, sample_weight
    
    def t_net_whole(self, all_point_set, all_semantic_seg):

        coordmax = np.max(all_point_set,axis=0) #find the max coord in whole list
        coordmin = np.min(all_point_set,axis=0)

        nsubvolume_x = np.ceil((coordmax[0]-coordmin[0])/2.5).astype(np.int32)
        nsubvolume_y = np.ceil((coordmax[1]-coordmin[1])/2.5).astype(np.int32)
        nsubvolume_x *= 2
        nsubvolume_y *= 2
        
        point_sets = list()
        semantic_segs = list()
        sample_weights = list()

        for i in range(nsubvolume_x):
            for j in range(nsubvolume_y):

                #define the cube by curmin and curmax
                curmin = coordmin+[i * 1.25, j * 1.25, 0] # moving cube 
                curmax = coordmin+[i * 1.25 + 2.5, j * 1.25 + 2.5, coordmax[2]-coordmin[2]]
    
                curchoice = np.sum((all_point_set >= (curmin-5))*(all_point_set <= (curmax+5)), axis=1) == 3

                cur_point_set = all_point_set[curchoice, :]
                cur_semantic_seg = all_semantic_seg[curchoice]

                if len(cur_semantic_seg) == 0:
                    continue

                # Select points between curmin and curmax but with smaller bound of curchoice
                mask = np.sum((cur_point_set >= (curmin-3))*(cur_point_set <= (curmax+3)), axis=1) == 3
                # Select number of point_sz as index from number of point found in curchoice
                choice = np.random.choice(len(cur_semantic_seg), self.point_sz, replace=True)
                
                point_set = cur_point_set[choice, :]  # Nx3
                semantic_seg = cur_semantic_seg[choice]  # N
                mask = mask[choice]
                
                if sum(mask) < 2000:
                    continue

                if sum(mask)/float(len(mask)) < 0.01:
                    # print("mask percentage: ", sum(mask)/float(len(mask)))
                    continue
                
                sample_weight = self.labelweights[semantic_seg]
                sample_weight *= mask  # N
                # print("sample_weight Unique:, ", np.unique(sample_weight))
                # print("********************")

                point_sets.append(np.expand_dims(point_set, 0))  # 1xNx3
                semantic_segs.append(np.expand_dims(semantic_seg, 0))  # 1xN
                sample_weights.append(np.expand_dims(sample_weight, 0))  # 1xN

        point_sets = np.concatenate(tuple(point_sets), axis=0)
        semantic_segs = np.concatenate(tuple(semantic_segs), axis=0)
        # print("semantic_segs unique: ",np.unique(semantic_segs))

        sample_weights = np.concatenate(tuple(sample_weights), axis=0)
        
        return point_sets, semantic_segs, sample_weights

    def t_net_evl(self, point_set, semantic_seg):

        coordmax = np.max(point_set,axis=0) #find the max coord in whole list
        coordmin = np.min(point_set,axis=0)

        #change the range of coordmax

        # smpmin = np.maximum(coordmax-[5, 5, 2], coordmin)

        # smpmin[2] = coordmin[2]
        # smpsz = np.minimum(coordmax-smpmin, [5, 5, 2])
        # smpsz[2] = coordmax[2]-coordmin[2]
        cur_semantic_seg = None
        cur_point_set = None

        for i in range(10):

            curcenter = point_set[np.random.choice(len(semantic_seg),1)[0],:] #random choice one point in the cube (get point index)

            curmin = curcenter-[2.5, 2.5, 1]
            curmax = curcenter+[2.5, 2.5, 1]
            curmin[2] = coordmin[2]
            curmax[2] = coordmax[2]
            curchoice = np.sum((point_set >= (curmin-1))*(point_set <= (curmax+1)),axis=1) == 3 #find the point in the cube (wider : 5)
            cur_point_set = point_set[curchoice, :]
            cur_semantic_seg = semantic_seg[curchoice]
            if len(cur_semantic_seg) == 0:
                continue
            mask = np.sum((cur_point_set >= (curmin-0.1))*(cur_point_set <= (curmax+0.1)), axis=1) == 3 #find the point in the cube (wider : 1)
            # cube has >= 70% valid label
            isvalid = np.sum(cur_semantic_seg > 0)/len(cur_semantic_seg) >= 0.7 
            if isvalid:
                break

        choice = np.random.choice(len(cur_semantic_seg), self.point_sz, replace=True)
        point_set = cur_point_set[choice,:]

        semantic_seg = cur_semantic_seg[choice]
        mask = mask[choice]
        sample_weight = self.labelweights[semantic_seg]
        sample_weight *= mask

        return point_set, semantic_seg, sample_weight
    
    def _augment_batch_data(self, batch_data):
        rotated_data = provider.rotate_point_cloud(batch_data)
        rotated_data = provider.rotate_perturbation_point_cloud(rotated_data)
        jittered_data = provider.random_scale_point_cloud(rotated_data[:,:,0:3])
        jittered_data = provider.shift_point_cloud(jittered_data)
        jittered_data = provider.jitter_point_cloud(jittered_data)
        rotated_data[:,:,0:3] = jittered_data
        return provider.shuffle_points(rotated_data)

    def load_h5_data(self, batch, position, mode):

        # for all data saved in h5
        all_data = h5py.File(os.path.join(DATA_PATH, TRAIN_FILE))

        # Calculate all class weights for all data at the begining
        if position == 0 and mode == 'train':
            print("Train mode")
            labelweights = np.zeros(8)
            for labels in all_data['base/category']:
                tmp, _ = np.histogram(labels, range(NUM_CLASS+1))
                labelweights += tmp
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights/np.sum(labelweights)
            labelweights = 1/np.log(1.2+labelweights)
            self.labelweights = labelweights
            print(self.labelweights)
            # exit()
        if mode == 'test':
            # print("Eval mode")
            self.labelweights = np.ones(NUM_CLASS)
            # print(self.labelweights)

        batch_data = np.zeros((batch, self.point_sz, 3))
        # batch_intensity = np.zeros((batch, self.point_sz, 1), dtype=np.float32)
        batch_label = np.zeros((batch, self.point_sz), dtype=np.int32)
        batch_smpw = np.zeros((batch, self.point_sz), dtype=np.float32)
        # for all data saved in h5
        _pts = all_data['base/pts'][position : position + batch]
        # _intensity = all_data['base/intensity'][position : position + batch]
        _category = all_data['base/category'][position : position + batch]

        #balanced data (shape do no math)
        # df = pd.DataFrame(data=np.transpose(_category)[:,:], index = None, columns = None)
        # df = df.apply(pd.Series.value_counts)

        # cate_0 = df.iloc[0] #find the max category appear frequency
        # cate_1_7 = df.iloc[1:].sum() #find the rest category appear frequency

        # drop_frame = cate_0/cate_1_7
        # drop_frame.index[drop_frame>100].tolist() #find which frame in batch should be dropped

        #shuffle data
        # print("_pts: ", _pts.shape)
        # print("_cate: ", _category.shape)
        # print("_cate_len: ", len(_category.shape))
        seed = np.random.permutation(_pts.shape[1])
        # print("b : ", seed.shape)
        # _pts, _intensity, _category = _pts[seed], _intensity[seed], _category[seed]
        _pts, _category = _pts[:,seed], _category[:,seed]

        # choice = self.t_net(_pts[1], _category[1]) #t-net

        #choice in batches so in 2nd -dim
        # _pts = _pts[:,choice,:]
        # _intensity = _intensity[:,choice]
        # _category = _category[:,choice]
        # _intensity = np.expand_dims(_intensity, 2)

        batch_data[:, ...] = _pts
        # batch_intensity[:, ...] = _intensity
        batch_label[:, :] = _category

        # Assign each single class in the batch data with corresponding class weight
        for class_num in range(8):
            # print(self.labelweights[i])
            batch_smpw[batch_label == class_num] = self.labelweights[class_num]

        return batch_data, batch_label, batch_smpw

    @staticmethod
    def ave_gradient(tower_grad):
        ave_gradient = []
        for gpu_data in zip(*tower_grad):
            grads = []
            for g, k in gpu_data:
                t_g = tf.expand_dims(g, axis=0)
                grads.append(t_g)
            grad = tf.concat(grads, axis=0)
            grad = tf.reduce_mean(grad, axis=0)
            key = gpu_data[0][1]
            ave_gradient.append((grad, key))
        return ave_gradient

    # cpu part of graph
    def build_g_cpu(self):
        self.batch = tf.Variable(0, name='batch', trainable=False)
        # self.point_pl, self.label_pl, self.intensity_pl = SEG_MODEL.placeholder_inputs(self.batch_sz, None) #placeholder shape!!
        self.point_pl, self.label_pl, self.smpws_pl = SEG_MODEL.placeholder_inputs(self.batch_sz, self.point_sz) #placeholder shape!!
        self.is_train_pl = tf.placeholder(dtype=tf.bool, shape=())
        self.ave_tp_pl = tf.placeholder(dtype=tf.float32, shape=())
        self.optimizer = tf.train.AdamOptimizer(self.get_learning_rate())
        self.bn_decay = self.get_bn_decay()
        # self.intensity_pl = tf.expand_dims(self.intensity_pl, axis = 2)
        SEG_MODEL.get_model(self.point_pl, self.is_train_pl, num_class=NUM_CLASS, bn_decay=self.bn_decay, feature = None)
    # graph for each gpu, reuse params...
    def build_g_gpu(self, gpu_idx):
        print("build graph in gpu %d" % gpu_idx)
        with tf.device('/gpu:%d' % gpu_idx), tf.name_scope('gpu_%d' % gpu_idx) as scope:
            point_cloud_slice = tf.slice(self.point_pl, [gpu_idx * BATCH_PER_GPU, 0, 0], [BATCH_PER_GPU, -1, -1])
            label_slice = tf.slice(self.label_pl, [gpu_idx * BATCH_PER_GPU, 0], [BATCH_PER_GPU, -1])
            smpws_slice = tf.slice(self.smpws_pl, [gpu_idx * BATCH_PER_GPU, 0], [BATCH_PER_GPU, -1])
            net, end_point = SEG_MODEL.get_model(point_cloud_slice, self.is_train_pl, num_class=NUM_CLASS,
                                                 bn_decay=self.bn_decay, feature = None)
            SEG_MODEL.get_loss(net, label_slice, smpw = smpws_slice) # change smpw to intensity
            loss = tf.get_collection('losses', scope=scope)
            total_loss = tf.add_n(loss, name='total_loss')
            for _i in loss + [total_loss]:
                tf.summary.scalar(_i.op.name, _i)

            gvs = self.optimizer.compute_gradients(total_loss)
            self.tower_grads.append(gvs)
            self.net_gpu.append(net)
            self.total_loss_gpu_list.append(total_loss)

    def build_graph(self):
        with tf.device('/cpu:0'):
            self.build_g_cpu()
            self.tower_grads = []
            self.net_gpu = []
            self.total_loss_gpu_list = []

            for i in range(GPU_NUM):
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    self.build_g_gpu(i)

            self.net = tf.concat(self.net_gpu, axis=0)
            self.loss = tf.reduce_mean(self.total_loss_gpu_list)

            # get training op
            gvs = self.ave_gradient(self.tower_grads)
            self.train_op = self.optimizer.apply_gradients(gvs, global_step=self.batch)
            self.predict = tf.cast(tf.argmax(self.net, axis=2), tf.int32)
            self.TP = tf.reduce_sum(
                tf.cast(tf.equal(self.predict, self.label_pl), tf.float32)) / self.batch_sz / self.point_sz
            tf.summary.scalar('TP', self.TP)
            tf.summary.scalar('total_loss', self.loss)

    def training(self):

        with tf.Graph().as_default():
            self.build_graph()
            # merge operator (for tensorboard)
            merged = tf.summary.merge_all()
            num_batches = self.train_sz // self.batch_sz ## TODO:  change train_sz
            saver = tf.train.Saver(max_to_keep = 250)

            # Create a session
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            config.log_device_placement = False
            best_acc = -1
            with tf.Session(config=config) as sess:
                train_writer = tf.summary.FileWriter(TRAIN_LOG_PATH, sess.graph)
                evaluate_writer = tf.summary.FileWriter(TEST_LOG_PATH, sess.graph)
                sess.run(tf.global_variables_initializer())
                epoch_sz = MAX_EPOCH
                tic = time.time()

            
                #load h5
                all_data = h5py.File(os.path.join(DATA_PATH, TRAIN_FILE))
                pts = all_data['base/pts'][:self.train_sz]
                category = all_data['base/category'][:self.train_sz].astype(int)

                # sample weights
                labelweights = np.zeros(8)
                # for labels in train_set['base/category']:
                for labels in category:
                    tmp, _ = np.histogram(labels, range(NUM_CLASS+1))
                    labelweights += tmp
                labelweights = labelweights.astype(np.float32)
                labelweights = labelweights/np.sum(labelweights)
                labelweights = 1/np.log(1.2+labelweights)
                self.labelweights = labelweights

                for epoch in range(epoch_sz):

                    ave_loss = 0

                    #shuffle data
                    train_idxs = np.arange(0, len(pts))
                    np.random.shuffle(train_idxs)

                    for _iter in tqdm(range(num_batches)):

                        start_idx = _iter * self.batch_sz
                        end_idx = (_iter+1) * self.batch_sz
                        batch_data, batch_label, batch_smpw = self.get_batch(pts, category, train_idxs, start_idx, end_idx)

                    # for position in tqdm(range(0, self.train_sz, self.batch_sz)):

                    # batch_data, batch_label, batch_smpw = self.load_h5_data(self.batch_sz, position, 'train')
                        # aug_data = self._augment_batch_data(batch_data)
                        aug_data = provider.rotate_point_cloud_z(batch_data)
                        loss, _, summary, step = sess.run([self.loss, self.train_op, merged, self.batch],
                                                          feed_dict={self.point_pl: aug_data,
                                                                     self.label_pl: batch_label,
                                                                     self.smpws_pl: batch_smpw,
                                                                     self.is_train_pl: True})
                        ave_loss += loss
                        train_writer.add_summary(summary, step)
                    ave_loss /= num_batches
                    print("epoch %d , loss is %f take %.3f s" % (epoch + 1, ave_loss, time.time() - tic))
                    tic = time.time()
                    if (epoch + 1) % 3 == 0:
                        acc = self.eval_one_epoch(sess, evaluate_writer, step, epoch)
                        # acc = self.eval_whole_one_epoch(sess, evaluate_writer, step, epoch)
                        if acc > best_acc:
                            _path = saver.save(sess, os.path.join(SAVE_PATH, "best_seg_model_%d.ckpt" % (epoch + 1)))
                            print("epoch %d, best saved in file: \n" % (epoch + 1), _path)
                            best_acc = acc
                        else:
                            _path = saver.save(sess, os.path.join(SAVE_PATH, "seg_model_%d.ckpt" % (epoch + 1)))
                            print("epoch %d, best saved in file: \n" % (epoch + 1), _path)
                    else:
                        acc = self.eval_one_epoch(sess, evaluate_writer, step, epoch)
                _path = saver.save(sess, os.path.join(SAVE_PATH, 'train_base_seg_model.ckpt'))
                print("Model saved in file: ", _path)

    def eval_one_epoch(self, sess, test_writer, step, epoch):
        """ ops: dict mapping from string to tf ops """

        is_training = False
        num_batches = self.test_sz//self.batch_sz

        loss_sum = 0
        IoU_sum = 0
        acc_sum = 0
        offset = self.train_sz
        print("---EVALUATE %d EPOCH---" % (epoch + 1))

        #load h5
        all_data = h5py.File(os.path.join(DATA_PATH, TRAIN_FILE))
        pts = all_data['base/pts'][self.train_sz : self.train_sz + self.test_sz]
        category = all_data['base/category'][self.train_sz : self.train_sz + self.test_sz].astype(int)

        #test index no shuffle
        test_idxs = np.arange(0, len(pts))

        #sample weight
        self.labelweights = np.ones(NUM_CLASS)

        for _iter in tqdm(range(num_batches)):

            start_idx = _iter * self.batch_sz
            end_idx = (_iter+1) * self.batch_sz
            batch_data, batch_label, batch_smpw = self.get_batch(pts, category, test_idxs, start_idx, end_idx)

        # for position in tqdm(range(0+offset, self.test_sz+self.train_sz, self.batch_sz)): #change no offset and train_sz

            # batch_data, batch_label, batch_smpw = self.load_h5_data(self.batch_sz, position, 'test') #in evl the weight shoulbe be all 1
            
            aug_data = provider.rotate_point_cloud_z(batch_data)
            # aug_data = self._augment_batch_data(batch_data)
            
            net, loss_val = sess.run([self.net, self.loss], feed_dict={self.point_pl: aug_data,
                                                                           self.label_pl: batch_label,
                                                                           self.smpws_pl: batch_smpw,
                                                                           self.is_train_pl: is_training})
            #loss and correction

            pred_val = np.argmax(net, axis=2)

            loss_sum += loss_val

            eps = 1e-6
            class_list = []
            accuracy_list = []

            for cate in range(1, NUM_CLASS):
                #if cate not in this frame
                if np.sum((batch_label == cate)) == 0:
                    # print("no cate <{}> in this frame ".format(cate))
                    continue
                print("--------------")
                print("label : ", cate)
                intersection = np.sum((pred_val == cate) & (batch_label == cate))
                print("gt : ", np.sum((batch_label == cate)))
                print("pred : ", np.sum((pred_val == cate)))
                print("intersection : ", intersection)
                
                union = np.sum(batch_label == cate) + np.sum(pred_val == cate) - intersection #union needs to minus intersection
                print("union: ", union)
                s_class = (intersection) / (union + eps)
                # print("s_class: ",s_class)
                class_list.append(s_class)

                accuracy = intersection / (np.sum(batch_label == cate) + eps)
                accuracy_list.append(accuracy)

            # print("stack class_list :", np.stack(class_list))
            IoU = np.mean(np.stack(class_list), 0)
            acc = np.mean(np.stack(accuracy_list), 0)
            print("mean :", IoU)
            
            IoU_sum += IoU
            acc_sum += acc

        print("Average accuracy : {0:.5f}".format(acc_sum/float(num_batches)))
        print("Average IoU : {0:.5f}".format(IoU_sum/float(num_batches)))
        print("Eval mean loss : {0:.5f}".format(loss_sum/float(num_batches)))
        return IoU_sum/float(num_batches)

    def eval_whole_one_epoch(self, sess, test_writer, step, epoch):
        """ ops: dict mapping from string to tf ops """

        is_training = False
        num_batches = self.test_sz//self.batch_sz

        loss_sum = 0
        IoU_sum = 0
        acc_sum = 0
        
        is_continue_batch = False

        extra_batch_data = np.zeros((0, self.point_sz, 3))
        extra_batch_label = np.zeros((0, self.point_sz))
        extra_batch_smpw = np.zeros((0, self.point_sz))
        batch_data, batch_label, batch_smpw = None, None, None
        print("---EVALUATE %d EPOCH---" % (epoch + 1))

        #load h5
        all_data = h5py.File(os.path.join(DATA_PATH, TRAIN_FILE))
        pts = all_data['base/pts'][self.train_sz : self.train_sz + self.test_sz]
        category = all_data['base/category'][self.train_sz : self.train_sz + self.test_sz].astype(int)

        #sample weight
        self.labelweights = np.ones(NUM_CLASS)
        
        for batch_idx in tqdm(range(num_batches)):
            print("batch_idx : ", batch_idx)
            if not is_continue_batch:
                batch_data, batch_label, batch_smpw = t_net_whole(pts[batch_idx,...], category[batch_idx,...])
                # print("batch_data : ", batch_data.shape)
                print(extra_batch_data)
                # print("extra_batch_data : ", extra_batch_data.shape)
                batch_data = np.concatenate((batch_data, extra_batch_data), axis=0)
                # print("after concate : ", batch_data.shape)
                batch_label = np.concatenate((batch_label, extra_batch_label), axis=0)
                batch_smpw = np.concatenate((batch_smpw, extra_batch_smpw), axis=0)
            else:
                batch_data_tmp, batch_label_tmp, batch_smpw_tmp = t_net_whole(pts[batch_idx,...], category[batch_idx,...])
                batch_data = np.concatenate((batch_data, batch_data_tmp), axis=0)
                batch_label = np.concatenate((batch_label, batch_label_tmp), axis=0)
                batch_smpw = np.concatenate((batch_smpw, batch_smpw_tmp), axis=0)
            if batch_data.shape[0] < self.batch_sz:
                # print("batch_data.shape[0] : ", batch_data.shape[0])
                is_continue_batch = True
                continue
            while batch_data.shape[0] >= self.batch_sz:
                is_continue_batch = False
                if batch_data.shape[0] == self.batch_sz:
                    extra_batch_data = np.zeros((0, self.point_sz, 3))
                    extra_batch_label = np.zeros((0, self.point_sz))
                    extra_batch_smpw = np.zeros((0, self.point_sz))
                else:
                    extra_batch_data = batch_data[self.batch_sz:, :, :]
                    extra_batch_label = batch_label[self.batch_sz:, :]
                    extra_batch_smpw = batch_smpw[self.batch_sz:, :]
                    batch_data = batch_data[: self.batch_sz, :, :]
                    batch_label = batch_label[: self.batch_sz, :]
                    batch_smpw = batch_smpw[: self.batch_sz, :]
                aug_data = batch_data
                net, loss_val = sess.run([self.net, self.loss], feed_dict={self.point_pl: aug_data,
                                                                           self.label_pl: batch_label,
                                                                           self.smpws_pl: batch_smpw,
                                                                           self.is_train_pl: is_training})
                pred_val = np.argmax(net, axis=2)
                loss_sum += loss_val

                eps = 1e-6
                class_list = []
                accuracy_list = []

                for cate in range(1, NUM_CLASS):
                    #if cate not in this frame
                    if np.sum((batch_label == cate)) == 0:
                        # print("no cate <{}> in this frame ".format(cate))
                        continue
                    print("--------------")
                    print("label : ", cate)
                    intersection = np.sum((pred_val == cate) & (batch_label == cate))
                    print("gt : ", np.sum((batch_label == cate)))
                    print("pred : ", np.sum((pred_val == cate)))
                    print("intersection : ", intersection)
                    
                    union = np.sum(batch_label == cate) + np.sum(pred_val == cate) - intersection #union needs to minus intersection
                    print("union: ", union)
                    s_class = (intersection) / (union + eps)
                    class_list.append(s_class)

                    accuracy = intersection / (np.sum(batch_label == cate) + eps)
                    accuracy_list.append(accuracy)

                batch_data = extra_batch_data
                batch_label = extra_batch_label
                batch_smpw = extra_batch_smpw
                
                #no label in this frame
                if len(class_list) > 0:

                    IoU = np.mean(np.stack(class_list), 0)
                    acc = np.mean(np.stack(accuracy_list), 0)
                    print("IoU :", IoU)
                
                    IoU_sum += IoU
                    acc_sum += acc

        print("Average accuracy : {0:.5f}".format(acc_sum/float(num_batches)))
        print("Average IoU : {0:.5f}".format(IoU_sum/float(num_batches)))
        print("Eval mean loss : {0:.5f}".format(loss_sum/float(num_batches)))

        return IoU_sum/float(num_batches)

if __name__ == '__main__':
    trainer = SegTrainer()
    trainer.training()
