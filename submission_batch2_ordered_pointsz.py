from __future__ import division
from __future__ import print_function

import os, sys
import numpy as np
import time
import tensorflow as tf
import argparse
import pandas as pd
from tqdm import tqdm
from itertools import chain

import tf_utils.provider as provider
import models.pointSIFT_pointnet as SEG_MODEL

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1, help='batch size during training[default: 32')
parser.add_argument('--root', default='/data/lecui/3d_drive', help='scannet dataset path')
parser.add_argument('--result_path', default='/data/lecui/3d_drive/Results', help='model param path')
parser.add_argument('--test_data_path', default='/data/lecui/3d_drive/TestSet', help='scannet dataset path')
parser.add_argument('--gpu_num', type=int, default=1, help='number of GPU to train')
#parser.add_argument('--model_path', default='/data/lecui/3d_drive/models/train_base_seg_model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')

FLAGS = parser.parse_args()
BATCH_SZ = FLAGS.batch_size
ROOT = FLAGS.root
RESULT_PATH = FLAGS.result_path
TEST_DATA_PATH = FLAGS.test_data_path
GPU_NUM = FLAGS.gpu_num
BATCH_PER_GPU = BATCH_SZ // GPU_NUM
point_sz = 5120
# MODEL_PATH = FLAGS.model_path
is_save_result = True #If save result ~!!!!

batch = 6000

MODEL_PATH = os.path.join(ROOT, 'Models_2/best_seg_model_200.ckpt') # choose the models
pts_file_path = os.path.join(ROOT, 'TestSet/pts/')
intensity_file_path = os.path.join(ROOT, 'TestSet/intensity/')
result_files = os.listdir(RESULT_PATH)
print("Model_path :" , MODEL_PATH)

# lr params..
DECAY_STEP = 200000
DECAY_RATE = 0.7

# bn params..
BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

def get_bn_decay():
    bn_momentum = tf.train.exponential_decay(BN_INIT_DECAY,
                                             batch * BATCH_SZ,
                                             BN_DECAY_DECAY_STEP,
                                             BN_DECAY_DECAY_RATE,
                                             staircase=True)
    bn_momentum = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    # tf.summary.scalar('bn_decay', bn_momentum)
    return bn_momentum

def _augment_batch_data(batch_data):
	rotated_data = provider.rotate_point_cloud(batch_data)
	rotated_data = provider.rotate_perturbation_point_cloud(rotated_data)
	jittered_data = provider.random_scale_point_cloud(rotated_data[:,:,0:3])
	jittered_data = provider.shift_point_cloud(jittered_data)
	jittered_data = provider.jitter_point_cloud(jittered_data)
	rotated_data[:,:,0:3] = jittered_data
	return provider.shuffle_points(rotated_data)

def test(save_mode = False):

    # is_training_pl = tf.constant(True)
    is_train_pl = tf.placeholder(dtype=tf.bool, shape=())

    # Creat placeholder
    point_pl, _ , intensity_pl = SEG_MODEL.placeholder_inputs(BATCH_SZ, None)
    intensity_pl = tf.expand_dims(intensity_pl, axis = 2)
    _bn_decay = get_bn_decay()
    # Load model
    net, end_points = SEG_MODEL.get_model(point_pl, is_train_pl, 8, bn_decay=_bn_decay, feature=None)

    # Create a saver
    saver = tf.train.Saver()

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False

    with tf.Session(config=config) as sess:
        # Restore variables from disk.
        saver.restore(sess, MODEL_PATH)
        print("Model restored.")
        print("Model_path :" , MODEL_PATH)

    #read csv path
        with open(os.path.join(ROOT, "test_pts.txt")) as f:
            txt = f.readlines()
            txt = list(map(lambda s: s.strip(), txt))[:batch] #get rid of "/n", if wanna batch change here !!!!!!
            print("\n# Total {} files need to be predicted !".format(len(txt)))
            done_txt = [file for file in txt if file in result_files]
            print("# Found {} files already been predicted in results file !".format(len(done_txt)))
            txt = [file for file in txt if file not in result_files] # not predict files
            print("# Found {} files waiting for predict".format(len(txt)))

        for i in tqdm(range(len(txt))):
            
            pts = pd.read_csv(os.path.join(pts_file_path, txt[i]), header = None, float_precision = 'round_trip')

            tic = time.time()
            inbound_pred_val = []

            x = 30
            y = 30
            z = 4
            box_x = 4
            box_y = 4
            stride = 4

            lidar_inbound = pts[(pts.iloc[:,0] < x) & (pts.iloc[:,0] > -x) & (pts.iloc[:,1] < y) & (pts.iloc[:,1] > -y) & (pts.iloc[:,2] < z) & (pts.iloc[:,2] > -z)]
            lidar_outbound = pts[(pts.iloc[:,0] >= x)|(pts.iloc[:,0] <= -x) | (pts.iloc[:,1] >= y) | (pts.iloc[:,1] <= -y) | (pts.iloc[:,2] >= z) | (pts.iloc[:,2] <= -z)]
            
            lidar_inbound_index = lidar_inbound.index
            lidar_outbound_index = lidar_outbound.index

            coordmax = np.max(np.array(lidar_inbound),axis=0) #find the max coord in whole list
            coordmin = np.min(np.array(lidar_inbound),axis=0)
            # print("coordmax : ", coordmax)
            # print("coordmin : ", coordmin)

            nsubvolume_x = np.ceil((coordmax[0]-coordmin[0])/box_x).astype(int)
            nsubvolume_y = np.ceil((coordmax[1]-coordmin[1])/box_y).astype(int)
            # print(nsubvolume_x)
            # print(nsubvolume_y)

            points_list = []
            true_list = []
            step_list = []
            curmin_max_list = []
         
            for _x in range(nsubvolume_x):
                for _y in range(nsubvolume_y):
                    curmin = coordmin + [_x * stride, _y * stride, 0]
                    curmax = coordmin + [_x * stride + box_x, _y * stride + box_y, coordmax[2]-coordmin[2]]
                    
                    curchoice = np.sum((lidar_inbound >= (curmin)) & (lidar_inbound <= (curmax)), axis=1) == 3
                    
                    if np.sum(curchoice) > 0: #box has points
                        _curchoice = curchoice[curchoice == True].index.tolist() #find points with index
                        points_list.append(_curchoice)

                    else:
                        continue
   
            order_lidar_index = list(set(chain.from_iterable(points_list))) #get rid off duplicate index
            if len(lidar_inbound) != len(order_lidar_index):
                _diff = set(lidar_inbound.index.tolist()) - set(order_lidar_index)
                # _diff = set(order_lidar_index) - set(lidar_inbound.index.tolist())
                print("diff index : ", _diff)
                print("differ point :" , lidar_inbound.loc[_diff].values)
                print("box lenght and frame length", len(lidar_inbound), len(order_lidar_index))
                print("coordmax : ", coordmax)
                print("coordmin : ", coordmin)
                print("the last curmin :", curmin)
                print("the last curmax :", curmax)
            
            pts = np.array(lidar_inbound.reindex(order_lidar_index)) #reorder index and covt to np.array
            
            for j in range(int(len(pts)/point_sz)): 
                _pts = np.expand_dims(pts[j * (point_sz) : j * (point_sz) + (point_sz) ,:], 0)
                # intensity = np.expand_dims(np.array(intensity.values)[:,:], 0)

                # aug_data = _augment_batch_data(pts) #data agument
                # aug_data = provider.rotate_point_cloud_z(_pts)
                aug_data = _pts

                _net = sess.run([net], feed_dict={point_pl: aug_data, is_train_pl : False})

                _net = np.array(_net[0])

                pred_val = np.argmax(_net, axis=2)

                inbound_pred_val.append(pred_val[0].tolist())

            if (len(pts) % point_sz) > 0:

                remain_pts = np.expand_dims(pts[-(len(pts) % point_sz) :,:], 0)
                # aug_data = provider.rotate_point_cloud_z(remain_pts)
                aug_data = remain_pts
                _net = sess.run([net], feed_dict={point_pl: aug_data, is_train_pl : False})
                _net = np.array(_net[0])
                pred_val = np.argmax(_net, axis=2)
                inbound_pred_val.append(pred_val[0].tolist())

            inbound_pred_val = list(chain.from_iterable(inbound_pred_val))
            inbound_pred_val = np.array(inbound_pred_val) #.reshape(-1,1)

            inbound_pred = pd.DataFrame(inbound_pred_val.astype(int), index = lidar_inbound_index)
            
            outbound_pred = pd.DataFrame(np.zeros(len(lidar_outbound)).astype(int), index = lidar_outbound_index)
            total_pred_val = np.array(pd.concat([inbound_pred, outbound_pred]).sort_index()) #!!!!!! sort index!
            # print("frame : %.3f s" % (time.time()- tic))

            if np.any(total_pred_val > 4)  :
                print('Find label 5 or 6 or 7')
                print(txt[i])
            if np.any(total_pred_val == 1)  :
                print('Find label 1')
                print(txt[i])
            if np.any(total_pred_val == 2)  :
                print('Find label 2')
                print(txt[i])
            # if np.any(total_pred_val > 0)  :
            #     print(np.unique(total_pred_val))
            #     print(txt[i])

            if save_mode == True:

                ##############################Double Check##########################################
                save_result(total_pred_val, os.path.join(RESULT_PATH, txt[i])) #result_path !!!!!!!!!
                ####################################################################################

def save_result(input, save_name):
    # input = input.transpose() # make row to column
    df = pd.DataFrame(data= input[:,:], index=None, dtype = np.int32).astype('str')
    df.to_csv(save_name, index = None,  header = False, encoding ='utf-8', line_terminator ='\n')

if __name__ == '__main__':

    with tf.Graph().as_default():
        test(save_mode = is_save_result)
