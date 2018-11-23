from __future__ import division
from __future__ import print_function

import os
import numpy as np
import time
import tensorflow as tf
import argparse
import h5py
from tqdm import tqdm

import tf_utils.provider as provider
import models.pointSIFT_pointnet as SEG_MODEL

parser = argparse.ArgumentParser()
parser.add_argument('--max_epoch', type=int, default=10, help='epoch to run[default: 1000]')
parser.add_argument('--batch_size', type=int, default=4, help='batch size during training[default: 32')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='initial learning rate[default: 1e-3]')
parser.add_argument('--save_path', default='/data/lecui/3d_drive/Models', help='model param path')
parser.add_argument('--data_path', default='/data/lecui/3d_drive/TrainSet', help='scannet dataset path')
parser.add_argument('--train_log_path', default='log/pointSIFT_train')
parser.add_argument('--test_log_path', default='log/pointSIFT_test')
parser.add_argument('--gpu_num', type=int, default=1, help='number of GPU to train')

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
        self.train_sz = 10000
        self.test_sz = 600
        self.point_sz = 12190 #5120

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

    def get_batch(self, dataset, idxs, start_idx, end_idx):
        bsize = end_idx - start_idx
        batch_data = np.zeros((bsize, self.point_sz, 3))
        batch_label = np.zeros((bsize, self.point_sz), dtype=np.int32)
        batch_intensity = np.zeros((bsize, self.point_sz, 1), dtype=np.float32)
        for i in range(bsize):
            ps, seg, intensity = dataset[idxs[i + start_idx]]
            batch_data[i, ...] = ps
            batch_label[i, :] = seg
            batch_intensity[i, ...] = intensity
        return batch_data, batch_label, batch_intensity

    def t_net(self, point_set, semantic_seg):

        coordmax = np.max(point_set,axis=0) #find the max coord in whole list
        coordmin = np.min(point_set,axis=0)

        smpmin = np.maximum(coordmax-[1.5, 1.5, 3.0], coordmin)

        smpmin[2] = coordmin[2]
        smpsz = np.minimum(coordmax-smpmin, [1.5, 1.5, 3.0])
        smpsz[2] = coordmax[2]-coordmin[2]
        cur_semantic_seg = None
        cur_point_set = None

        for i in range(10):
            curcenter = point_set[np.random.choice(len(semantic_seg),1)[0],:]
            curmin = curcenter-[0.75, 0.75, 1.5]
            curmax = curcenter+[0.75, 0.75, 1.5]
            curmin[2] = coordmin[2]
            curmax[2] = coordmax[2]
            curchoice = np.sum((point_set >= (curmin-0.2))*(point_set <= (curmax+0.2)),axis=1) == 3
            cur_point_set = point_set[curchoice, :]
            cur_semantic_seg = semantic_seg[curchoice]
            if len(cur_semantic_seg) == 0:
                continue
            mask = np.sum((cur_point_set >= (curmin-0.01))*(cur_point_set <= (curmax+0.01)), axis=1) == 3
            vidx = np.ceil((cur_point_set[mask, :]-curmin)/(curmax-curmin)*[31.0, 31.0, 62.0])
            vidx = np.unique(vidx[:, 0] * 31.0 * 62.0 + vidx[:, 1] * 62.0 + vidx[:, 2])
            isvalid = np.sum(cur_semantic_seg > 0)/len(cur_semantic_seg) >= 0.7 and len(vidx)/31.0/31.0/62.0 >= 0.02
            if isvalid:
                break
        choice = np.random.choice(len(cur_semantic_seg), self.point_sz, replace=True)

        return choice

    def load_h5_data(self, batch, position):

        # for all data saved in h5
        # all_data = h5py.File(os.path.join(DATA_PATH, "train_all_data.h5"))

        pts = h5py.File(os.path.join(DATA_PATH, "pts_3dims.h5"))
        intensity = h5py.File(os.path.join(DATA_PATH, "intensity_3dims.h5"))
        category = h5py.File(os.path.join(DATA_PATH, "category_3dims.h5"))

        batch_data = np.zeros((batch, self.point_sz, 3))
        batch_intensity = np.zeros((batch, self.point_sz, 1), dtype=np.float32)
        batch_label = np.zeros((batch, self.point_sz), dtype=np.int32)

        _pts = pts['pts'][position : position + batch]
        _intensity = intensity['intensity'][position : position + batch]
        _category = category['category'][position : position + batch]

        # # for all data saved in h5
        # _pts = all_data['base/pts'][position : position + batch]
        # _intensity = all_data['base/intensity'][position : position + batch]
        # _category = all_data['base/category'][position : position + batch]

        #shuffle data
        seed = np.random.permutation(_pts.shape[0])
        _pts, _intensity, _category = _pts[seed], _intensity[seed], _category[seed]
        #print(_category.shape)
        _category = _category[:,:,0] #only for seperate h5
    
        choice = self.t_net(_pts[0], _category[1]) #t-net KNN

        #choice in batches so in 2nd -dim
        _pts = _pts[:,choice,:]
        _intensity = _intensity[:,choice, :]
        _category = _category[:,choice]

        batch_data[:, ...] = _pts
        batch_intensity[:, ...] = _intensity
        batch_label[:, :] = _category

        return batch_data, batch_label, batch_intensity

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
        self.point_pl, self.label_pl, self.intensity_pl = SEG_MODEL.placeholder_inputs(self.batch_sz, None) #placeholder shape!!
        self.is_train_pl = tf.placeholder(dtype=tf.bool, shape=())
        self.ave_tp_pl = tf.placeholder(dtype=tf.float32, shape=())
        self.optimizer = tf.train.AdamOptimizer(self.get_learning_rate())
        self.bn_decay = self.get_bn_decay()
        self.intensity_pl = tf.expand_dims(self.intensity_pl, axis = 2)
        SEG_MODEL.get_model(self.point_pl, self.is_train_pl, num_class=NUM_CLASS, bn_decay=self.bn_decay, feature = self.intensity_pl)
    # graph for each gpu, reuse params...
    def build_g_gpu(self, gpu_idx):
        print("build graph in gpu %d" % gpu_idx)
        with tf.device('/gpu:%d' % gpu_idx), tf.name_scope('gpu_%d' % gpu_idx) as scope:
            point_cloud_slice = tf.slice(self.point_pl, [gpu_idx * BATCH_PER_GPU, 0, 0], [BATCH_PER_GPU, -1, -1])
            label_slice = tf.slice(self.label_pl, [gpu_idx * BATCH_PER_GPU, 0], [BATCH_PER_GPU, -1])
            intensity_slice = tf.slice(self.intensity_pl, [gpu_idx * BATCH_PER_GPU, 0, 0], [BATCH_PER_GPU, -1, -1])
            net, end_point = SEG_MODEL.get_model(point_cloud_slice, self.is_train_pl, num_class=NUM_CLASS,
                                                 bn_decay=self.bn_decay, feature = intensity_slice)
            SEG_MODEL.get_loss(net, label_slice, smpw=intensity_slice) # change smpw to intensity
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
            iter_in_epoch = self.train_sz // self.batch_sz ## TODO:  change train_sz
            saver = tf.train.Saver()

            # Create a session
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            config.log_device_placement = False
            best_acc = 0.0
            with tf.Session(config=config) as sess:
                train_writer = tf.summary.FileWriter(TRAIN_LOG_PATH, sess.graph)
                evaluate_writer = tf.summary.FileWriter(TEST_LOG_PATH, sess.graph)
                sess.run(tf.global_variables_initializer())
                epoch_sz = MAX_EPOCH
                tic = time.time()

                for epoch in range(epoch_sz):
                    ave_loss = 0
                    for position in tqdm(range(0, self.train_sz, self.batch_sz)):

                        batch_data, batch_label, batch_intensity = self.load_h5_data(self.batch_sz, position)
                        
                        aug_data = provider.rotate_point_cloud_z(batch_data)
                        loss, _, summary, step = sess.run([self.loss, self.train_op, merged, self.batch],
                                                          feed_dict={self.point_pl: aug_data,
                                                                     self.label_pl: batch_label,
                                                                     self.intensity_pl: batch_intensity,
                                                                     self.is_train_pl: True})
                        ave_loss += loss
                        train_writer.add_summary(summary, step)
                    ave_loss /= iter_in_epoch
                    print("epoch %d , loss is %f take %.3f s" % (epoch + 1, ave_loss, time.time() - tic))
                    tic = time.time()
                    if (epoch + 1) % 2 == 0:
                        acc = self.eval_one_epoch(sess, evaluate_writer, step, epoch)
                        if acc > best_acc:
                            _path = saver.save(sess, os.path.join(SAVE_PATH, "best_seg_model_%d.ckpt" % (epoch + 1)))
                            print("epoch %d, best saved in file: \n" % (epoch + 1), _path)
                            best_acc = acc
                _path = saver.save(sess, os.path.join(SAVE_PATH, 'train_base_seg_model.ckpt'))
                print("Model saved in file: ", _path)

    def eval_one_epoch(self, sess, test_writer, step, epoch):
        """ ops: dict mapping from string to tf ops """

        is_training = False
        num_batches = self.test_sz//self.batch_sz

        loss_sum = 0
        IoU_sum = 0
        offset = self.train_sz
        print("---EVALUATE %d EPOCH---" % (epoch + 1))

        for position in tqdm(range(0+offset, self.train_sz + self.test_sz, self.batch_sz)):
            
            batch_data, batch_label, batch_intensity = self.load_h5_data(self.batch_sz, position)

            aug_data = provider.rotate_point_cloud_z(batch_data)

            net, loss_val = sess.run([self.net, self.loss], feed_dict={self.point_pl: aug_data,
                                                                           self.label_pl: batch_label,
                                                                           self.intensity_pl: batch_intensity,
                                                                           self.is_train_pl: is_training})
            #loss and correction

            pred_val = np.argmax(net, axis=2)
            print("label : ", np.where(pred_val > 0))
            loss_sum += loss_val

            eps = 1e-6
            class_list = []
            for cate in range(1, NUM_CLASS-1):
                intersection = np.sum((pred_val == cate) & (batch_label == cate))
                union = np.sum(batch_label == cate) + np.sum(pred_val == cate)
                s_class = (intersection + eps) / (union + eps)
                class_list.append(s_class)

            IoU = np.mean(np.stack(class_list), 0)
            IoU_sum += IoU

        print("Average IoU : {0:.5f}".format(IoU_sum/float(num_batches)))
        print("Eval mean loss : {0:.5f}".format(loss_sum/float(num_batches)))

        return IoU_sum/float(num_batches)

if __name__ == '__main__':
    trainer = SegTrainer()
    trainer.training()
