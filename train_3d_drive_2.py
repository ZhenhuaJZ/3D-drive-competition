from __future__ import division
from __future__ import print_function

import os,sys
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
parser.add_argument('--batch_size', type=int, default=8, help='batch size during training[default: 32')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='initial learning rate[default: 1e-3]')
parser.add_argument('--save_path', default='/data/lecui/3d_drive/Models_2', help='model param path')
parser.add_argument('--data_path', default='/data/lecui/3d_drive/TrainSet', help='scannet dataset path')
parser.add_argument('--train_log_path', default='/data/lecui/3d_drive/log/pointSIFT_train_2')
parser.add_argument('--test_log_path', default='/data/lecui/3d_drive/log/pointSIFT_test_2')
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
TRAIN_FILE = 'train_r5.h5' #'train_r5 --> total 4775

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
        self.train_sz = 4000
        self.test_sz = 100
        self.point_sz = 5120 #8192 #5120

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

            dropout_ratio = np.random.random() * 0.875  # 0-0.875
            drop_idx = np.where(np.random.random((pts.shape[0])) <= dropout_ratio)[0]

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

            _pts, _category, _smpw = self.t_net_dirct_choose(_pts, _category) #t-net

            batch_data[i, ...] = _pts
            batch_label[i, :] = _category
            batch_smpw[i, :] = _smpw

            # for class_num in range(8):
                # batch_smpw[batch_label == class_num] = self.labelweights[class_num]

        return batch_data, batch_label, batch_smpw   

    def t_net_dirct_choose(self, point_set, semantic_seg):

            # print("frame contains semantic_seg : ", np.unique(semantic_seg))
            label_in_frame = np.unique(semantic_seg)
            coordmax = np.max(point_set,axis=0) #find the max coord in whole list
            coordmin = np.min(point_set,axis=0)

            x = 30
            y = 30
            z = 4 # max box height
            if coordmax[2] > z: coordmax[2] = z
            if coordmin[2] < -z: coordmin[2] = -z

            box_x = 4
            box_y = 4

            bounder_x = x-box_x/2

            #change the range of coordmax
            cur_semantic_seg = None
            cur_point_set = None
            mask = None
            target_label = np.random.choice(label_in_frame[1:],1)
            target_label_ratio = np.sum(semantic_seg == target_label)/len(semantic_seg)
            # print("target_label :", target_label) #random choice label except 0  ([1:])
            # print("target_label_ratio : ", target_label_ratio)

            for i in range(300):

                point_index = np.random.choice(len(semantic_seg),1)[0]
                curcenter = point_set[point_index,:] #random choice one point in the cube (get point index)
                curcenter_label = semantic_seg[point_index]

                # range condition (make suer the random point restrict in a certain range)
                while np.any(curcenter > bounder_x) or np.any(curcenter < -bounder_x) or curcenter[2] <= -z or curcenter[2] > z:
                    point_index = np.random.choice(len(semantic_seg),1)[0]
                    curcenter = point_set[point_index,:]
                    curcenter_label = semantic_seg[point_index] 

                # a box that slide the z's height 
                curmin = curcenter-[box_x/2, box_y/2, 1]
                curmax = curcenter+[box_x/2, box_y/2, 1]
                curmin[2] = coordmin[2]
                curmax[2] = coordmax[2]

                curchoice = np.sum((point_set >= (curmin-0.1))*(point_set <= (curmax+0.1)),axis=1) == 3 #find the point in the cube (wider : 5)
                cur_point_set = point_set[curchoice, :]
                cur_semantic_seg = semantic_seg[curchoice]
                if len(cur_semantic_seg) == 0:
                    continue
                mask = np.sum((cur_point_set >= (curmin-0.01))*(cur_point_set <= (curmax+0.01)), axis=1) == 3 #find the point in the cube (wider : 1)

                # cube has >= 70% valid label
                isvalid = np.sum(cur_semantic_seg == curcenter_label)/len(cur_semantic_seg) >= target_label_ratio and curcenter_label == target_label
                if isvalid:
                    break

            choice = np.random.choice(len(cur_semantic_seg), self.point_sz, replace=True)
            point_set = cur_point_set[choice,:]

            semantic_seg = cur_semantic_seg[choice]
            # print("choice contain semantic_seg label : {} \n".format(np.unique(semantic_seg)))
            mask = mask[choice]
            sample_weight = self.labelweights[semantic_seg]
            sample_weight *= mask
          

            return point_set, semantic_seg, sample_weight

    def t_net(self, point_set, semantic_seg):

        coordmax = np.max(point_set,axis=0) #find the max coord in whole list
        coordmin = np.min(point_set,axis=0)

        x = 60
        y = 60
        z = 4.5 # max box height
        if coordmax[2] > z: coordmax[2] = z
        if coordmin[2] < -z: coordmin[2] = -z

        box_x = 5
        box_y = 5

        bounder_x = x-box_x/2

        #change the range of coordmax
        cur_semantic_seg = None
        cur_point_set = None
        mask = None

        for i in range(90):

            curcenter = point_set[np.random.choice(len(semantic_seg),1)[0],:] #random choice one point in the cube (get point index)

            # range condition (make suer the random point restrict in a certain range)
            while np.any(curcenter > bounder_x) or np.any(curcenter < -bounder_x) or curcenter[2] <= -z or curcenter[2] > z:
                curcenter = point_set[np.random.choice(len(semantic_seg),1)[0],:]

            # a box that slide the z's height 
            curmin = curcenter-[box_x/2, box_y/2, 1]
            curmax = curcenter+[box_x/2, box_y/2, 1]
            curmin[2] = coordmin[2]
            curmax[2] = coordmax[2]

            curchoice = np.sum((point_set >= (curmin-0.1))*(point_set <= (curmax+0.1)),axis=1) == 3 #find the point in the cube (wider : 5)
            cur_point_set = point_set[curchoice, :]
            cur_semantic_seg = semantic_seg[curchoice]
            if len(cur_semantic_seg) == 0:
                continue
            mask = np.sum((cur_point_set >= (curmin-0.01))*(cur_point_set <= (curmax+0.01)), axis=1) == 3 #find the point in the cube (wider : 1)

            # cube has >= 70% valid label
            isvalid = np.sum(cur_semantic_seg > 0)/len(cur_semantic_seg) >= 0.5
            if isvalid:
                break

        choice = np.random.choice(len(cur_semantic_seg), self.point_sz, replace=True)
        point_set = cur_point_set[choice,:]

        semantic_seg = cur_semantic_seg[choice]
        mask = mask[choice]
        sample_weight = self.labelweights[semantic_seg]
        sample_weight *= mask

        return point_set, semantic_seg, sample_weight
    
    def t_net_whole(self, all_point_set, all_semantic_seg):

        coordmax = np.max(all_point_set,axis=0) #find the max coord in whole list
        coordmin = np.min(all_point_set,axis=0)

        bound_x = 50
        bound_y = 50 
        bound_z = 4 # box max height

        #searching stride
        stride = 4 # stride = box_x  -- means no overlapping
        box_x = 4
        box_y = 4

        if coordmax[0] > bound_x: coordmax[0] = bound_x
        if coordmax[1] > bound_y: coordmax[1] = bound_y
        if coordmax[2] > bound_z: coordmax[2] = bound_z

        if coordmin[0] < -bound_x: coordmin[0] = -bound_x
        if coordmin[1] < -bound_y: coordmin[1] = -bound_y
        if coordmin[2] < -bound_z: coordmin[2] = -bound_z

        nsubvolume_x = np.ceil((coordmax[0]-coordmin[0])/box_x).astype(np.int32) #x grid
        nsubvolume_y = np.ceil((coordmax[1]-coordmin[1])/box_y).astype(np.int32) #y grid
        
        point_sets = list()
        semantic_segs = list()
        sample_weights = list()

        # print("runing : {} * {} = {} times".format(nsubvolume_x, nsubvolume_y, nsubvolume_x * nsubvolume_y))

        for i in range(nsubvolume_x):
            for j in range(nsubvolume_y):

                #define the cube by curmin and curmax
                curmin = coordmin+[i * stride, j * stride, 0] # moving cube 
                curmax = coordmin+[i * stride + box_x, j * stride + box_y, coordmax[2]-coordmin[2]] #coordmax[2]-coordmin[2]
    
                curchoice = np.sum((all_point_set >= (curmin-0.1))*(all_point_set <= (curmax+0.1)), axis=1) == 3

                cur_point_set = all_point_set[curchoice, :]
                cur_semantic_seg = all_semantic_seg[curchoice]

                if len(cur_semantic_seg) == 0:
                    continue
                # Select points between curmin and curmax but with smaller bound of curchoice
                mask = np.sum((cur_point_set >= (curmin-0.01))*(cur_point_set <= (curmax+0.01)), axis=1) == 3
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

                point_sets.append(np.expand_dims(point_set, 0))  # 1xNx3
                semantic_segs.append(np.expand_dims(semantic_seg, 0))  # 1xN
                sample_weights.append(np.expand_dims(sample_weight, 0))  # 1xN

        point_sets = np.concatenate(tuple(point_sets), axis=0)
        semantic_segs = np.concatenate(tuple(semantic_segs), axis=0)
        # print("semantic_segs unique: ",np.unique(semantic_segs))
        # print("semantic_segs sum:, ", np.sum(semantic_segs > 0)) # how many labels overlaped
        # print("semantic_segs shape : ", semantic_segs.shape)
        sample_weights = np.concatenate(tuple(sample_weights), axis=0)
        

        return point_sets, semantic_segs, sample_weights

    def _augment_batch_data(self, batch_data):
        rotated_data = provider.rotate_point_cloud(batch_data)
        rotated_data = provider.rotate_perturbation_point_cloud(rotated_data)
        jittered_data = provider.random_scale_point_cloud(rotated_data[:,:,0:3])
        jittered_data = provider.shift_point_cloud(jittered_data)
        jittered_data = provider.jitter_point_cloud(jittered_data)
        rotated_data[:,:,0:3] = jittered_data
        return provider.shuffle_points(rotated_data)

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
            # SEG_MODEL.get_loss(net, label_slice, smpw = smpws_slice)
            SEG_MODEL.get_focal_loss(net, label_slice)
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
                print("\n# Start with {} frames data {} training and {} testing".format(all_data['base/category'].shape[0], self.train_sz, self.test_sz))
                pts = all_data['base/pts'][:self.train_sz]
                category = all_data['base/category'][:self.train_sz].astype(int)
                # print("all: ", np.sum(category > 0))
                # print("1: ", np.sum(category == 1))
                # print("2: ", np.sum(category == 2))
                # print("3: ", np.sum(category == 3))
                # print("4: ", np.sum(category == 4))
                # print("5: ", np.sum(category == 5))
                # print("6: ", np.sum(category == 6))
                # print("7: ", np.sum(category == 7))
                # sample weights
                labelweights = np.zeros(NUM_CLASS)
                # for labels in train_set['base/category']:
                for labels in category:
                    # print("labels is : ", np.unique(labels))
                    # print("labels_2 is : ", np.unique(np.delete(labels,np.where(labels == 0))))
                    # weightr the label without 0
                    labels = np.delete(labels,np.where(labels == 0))
                    tmp, _ = np.histogram(labels, range(NUM_CLASS+1))
                    # tmp, _ = np.histogram(labels, range(NUM_CLASS+1))
                    labelweights += tmp
                labelweights = labelweights.astype(np.float32)
                labelweights = labelweights/np.sum(labelweights)
                labelweights = 1/np.log(1.2+labelweights)
                labelweights[0] = 1
                self.labelweights = labelweights

                for epoch in range(epoch_sz):

                    ave_loss = 0

                    #shuffle data
                    train_idxs = np.arange(0, len(pts))
                    np.random.shuffle(train_idxs)
                    print(train_idxs)

                    for _iter in tqdm(range(num_batches)):

                        start_idx = _iter * self.batch_sz
                        end_idx = (_iter+1) * self.batch_sz
                        batch_data, batch_label, batch_smpw = self.get_batch(pts, category, train_idxs, start_idx, end_idx)

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
                    if (epoch + 1) % 5 == 0:
                        # acc = self.eval_one_epoch(sess, evaluate_writer, step, epoch)
                        acc = self.eval_whole_one_epoch(sess, evaluate_writer, step, epoch)
                        if acc > best_acc:
                            _path = saver.save(sess, os.path.join(SAVE_PATH, "best_seg_model_%d.ckpt" % (epoch + 1)))
                            print("epoch %d, best saved in file: \n" % (epoch + 1), _path)
                            best_acc = acc
                        else:
                            _path = saver.save(sess, os.path.join(SAVE_PATH, "seg_model_%d.ckpt" % (epoch + 1)))
                            print("epoch %d, saved in file: \n" % (epoch + 1), _path)
                    # else:
                        # acc = self.eval_one_epoch(sess, evaluate_writer, step, epoch)
                        # acc = self.eval_whole_one_epoch(sess, evaluate_writer, step, epoch)
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
                    continue
                intersection = np.sum((pred_val == cate) & (batch_label == cate))
                union = np.sum(batch_label == cate) + np.sum(pred_val == cate) - intersection #union needs to minus intersection
                s_class = (intersection) / (union + eps)
                class_list.append(s_class)

                accuracy = intersection / (np.sum(batch_label == cate) + eps)
                accuracy_list.append(accuracy)

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
        eps = 1e-6

        loss_sum = 0
        IoU_sum = 0
        acc_sum = 0
        total_correct = 0
        total_seen = 0
        total_seen_class = [0 for _ in range(NUM_CLASS)]
        total_correct_class = [0 for _ in range(NUM_CLASS)]
        total_iou = [0 for _ in range(1, NUM_CLASS)]
        class_iou_sum = []
        
        is_continue_batch = False

        extra_batch_data = np.zeros((0, self.point_sz, 3))
        extra_batch_label = np.zeros((0, self.point_sz))
        extra_batch_smpw = np.zeros((0, self.point_sz))
        batch_data, batch_label, batch_smpw = None, None, None
        print("\n---EVALUATE %d EPOCH---" % (epoch + 1))

        #load h5
        all_data = h5py.File(os.path.join(DATA_PATH, TRAIN_FILE))
        pts = all_data['base/pts'][self.train_sz : self.train_sz + self.test_sz]
        category = all_data['base/category'][self.train_sz : self.train_sz + self.test_sz].astype(int)
        print("Validation set contains labels : ", np.unique(category))

        #sample weight
        self.labelweights = np.ones(NUM_CLASS)

        for frame_index in tqdm(range(self.test_sz)):

            IoU_each_frame = []
            acc_each_frame = []
            loss_each_frame =[]

            if not is_continue_batch:
            	#(overlap, point_sz,3)
                batch_data, batch_label, batch_smpw = self.t_net_whole(pts[frame_index,...], category[frame_index,...])
                batch_data = np.concatenate((batch_data, extra_batch_data), axis=0)
                batch_label = np.concatenate((batch_label, extra_batch_label), axis=0)
                batch_smpw = np.concatenate((batch_smpw, extra_batch_smpw), axis=0)
            else:
                batch_data_tmp, batch_label_tmp, batch_smpw_tmp = self.t_net_whole(pts[frame_index,...], category[frame_index,...])
                batch_data = np.concatenate((batch_data, batch_data_tmp), axis=0)
                batch_label = np.concatenate((batch_label, batch_label_tmp), axis=0)
                batch_smpw = np.concatenate((batch_smpw, batch_smpw_tmp), axis=0)
            if batch_data.shape[0] < self.batch_sz:
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
                loss_each_frame.append(loss_val)
                # correct = np.sum((pred_val == batch_label) & (batch_label > 0) & (batch_smpw > 0)) #smpw > 0 ? (intersection)
                # total_correct += correct
                # total_seen += np.sum((batch_label > 0) & (batch_smpw > 0))
    
                # for l in range(NUM_CLASS):

                #     total_seen_class[l] += np.sum((batch_label == l) & (batch_smpw > 0))
                #     total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l) & (batch_smpw > 0))
                
                class_list = []
                accuracy_list = []

                for l in range(1, NUM_CLASS):
                    #if no cate in this frame then break the loop
                    if np.sum(batch_label) == 0:
                        # print("no cate <{}> in this frame ".format(l))
                        break
                    intersection = np.sum((pred_val == l) & (batch_label == l))
                    union = np.sum(batch_label == l) + np.sum(pred_val == l) - intersection #union needs to minus intersection

                    iou = (intersection) / (union + eps)
                    accuracy = intersection / (np.sum(batch_label == l) + eps)
                    class_list.append(iou)
                    accuracy_list.append(accuracy)
                    #each class iou
                    total_iou[l-1] += iou

                    # print("gt : ", np.sum((batch_label == l)))
                    # print("pred : ", np.sum((pred_val == l)))


                #no label in this frame
                if len(class_list) != 0:

                    IoU = np.mean(np.stack(class_list), 0)
                    acc = np.mean(np.stack(accuracy_list), 0)
                    IoU_each_frame.append(IoU)
                    acc_each_frame.append(acc)

                batch_data = extra_batch_data
                batch_label = extra_batch_label
                batch_smpw = extra_batch_smpw
            
            #evl details for each frame (adding all the frame together)
            IoU_sum += sum(IoU_each_frame)/len(IoU_each_frame)
            acc_sum += sum(acc_each_frame)/len(acc_each_frame)
            loss_sum += sum(loss_each_frame)/len(loss_each_frame)
            class_iou_sum.append(np.array(total_iou)/len(IoU_each_frame))

        #evl details for total
        total_iou = np.mean(np.stack(class_iou_sum), axis = 0)
        print("Average accuracy : {0:.5f}".format(acc_sum/self.test_sz))
        print("Average IoU : {0:.5f}".format(IoU_sum/self.test_sz))
        print("Eval mean loss : {0:.5f}".format(loss_sum/self.test_sz))
        # print('Eval whole scene point accuracy: %f' % (total_correct / float(total_seen)))
        per_class_str = 'points cloud base --------'
        for l in range(1, NUM_CLASS):
            per_class_str += 'class %d , iou: %f; ' % (l, (total_iou/self.test_sz)[l-1])
        print(per_class_str)

        return IoU_sum/self.test_sz

if __name__ == '__main__':
    trainer = SegTrainer()
    trainer.training()
