from __future__ import division
from __future__ import print_function

import os
import numpy as np
import time
import tensorflow as tf
import argparse
import pandas as pd
from tqdm import tqdm

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
# MODEL_PATH = FLAGS.model_path
is_save_result = True #If save result ~!!!!
batch = 6000

MODEL_PATH = os.path.join(ROOT, 'Models_2/12_10_18/seg_model_10.ckpt') # choose the models
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
    # intensity_pl = tf.expand_dims(intensity_pl, axis = 2)
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
    # with tf.device('/gpu:1'):
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
            done_txt = [f for f in txt if f in result_files]
            print("# Found {} files already been predicted in results file !".format(len(done_txt)))
            txt = [f for f in txt if f not in result_files] # not predict files
            print("# Found {} files waiting for predict".format(len(txt)))

        for i in tqdm(range(len(txt))):
            pts = pd.read_csv(os.path.join(pts_file_path, txt[i]), header = None)
            # print("text file: ", txt[i])
            # print("z max", pts.iloc[:,2].max())
            # print("z min", pts.iloc[:,2].min())
            # print("number of between -4 to 4 meters: ", len(pts[(pts.iloc[:,2] > -4) & (pts.iloc[:,2] < 4)]))
            # print("number of outside -4 meters: ", len(pts[(pts.iloc[:,2] <= -4) | (pts.iloc[:,2] >= 4)]))
            # print("total number: ", len(pts.iloc[:,2]))
            # print(sum(pts.iloc[:,2] > 8)/float(len(pts.iloc[:,2])))
            # print('****************')
            # intensity = pd.read_csv(os.path.join(intensity_file_path, txt[i]), header = None)
            pts = np.expand_dims(np.array(pts.values)[:,:], 0)
            # intensity = np.expand_dims(np.array(intensity.values)[:,:], 0)

            # aug_data = _augment_batch_data(pts) #data agument
            aug_data = provider.rotate_point_cloud_z(pts)

            tic = time.time()
            
            _net = sess.run([net], feed_dict={point_pl: aug_data, is_train_pl : False})
            
            _net = np.array(_net[0])
       
            pred_val = np.argmax(_net, axis=2)
            # print("frame : %.3f s" % (time.time()- tic))
            # if len(np.unique(pred_val)) > 1:
            #     print(np.unique(pred_val))
            #     print(txt[i])
  
            if save_mode == True:
                ##############################Double Check##########################################
                save_result(pred_val, os.path.join(RESULT_PATH, txt[i])) #result_path !!!!!!!!!
                ####################################################################################

def save_result(input, save_name):
    input = input.transpose() # make row to column
    df = pd.DataFrame(data= input[:,:], index=None, dtype = np.int32).astype('str')
    df.to_csv(save_name, index = None,  header = False, encoding ='utf-8', line_terminator ='\n')

if __name__ == '__main__':

    with tf.Graph().as_default():
        test(save_mode = is_save_result)
