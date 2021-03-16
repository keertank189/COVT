
import numpy as np
import cv2
import json
import os
import sys
import glob
import copy
import pickle
import scipy.misc as sic



import tensorflow as tf
import tensorflow.contrib.slim as slim
from lib.model import data_loader, generator, SRGAN, test_data_loader, inference_data_loader, save_images, SRResnet
from lib.ops import *
import math
import time 

name = ""
inputs_raw = tf.placeholder(tf.float32, shape=[1, None, None, 3], name='inputs_raw')
path_LR = tf.placeholder(tf.string, shape=[], name='path_LR')
scal = tf.placeholder(tf.int32, shape = [], name = 'scal')

with tf.variable_scope('generator'):
	gen_output = generator(inputs_raw, 3, tf.AUTO_REUSE,  nores=16)
	gen_output = tf.cond(tf.equal(scal, tf.constant(4)), lambda: generator(deprocess(gen_output), 3, tf.AUTO_REUSE,  nores=16), lambda: gen_output)
	
	print('Finish building the network')
	with tf.name_scope('convert_image'):
		# Deprocess the images outputed from the model
		inputs = deprocessLR(inputs_raw)
		outputs = deprocess(gen_output)
		# Convert back to uint8
		converted_inputs = tf.image.convert_image_dtype(inputs, dtype=tf.uint8, saturate=True)
		converted_outputs = tf.image.convert_image_dtype(outputs, dtype=tf.uint8, saturate=True)

with tf.name_scope('encode_image'):
	save_fetch = {
		"path_LR": path_LR,
		"inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name='input_pngs'),
		"outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name='output_pngs')
	}

# Define the weight initiallizer (In inference time, we only need to restore the weight of the generator)
var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
weight_initiallizer = tf.train.Saver(var_list)

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():

	# Define the initialization operation
	init_op = tf.global_variables_initializer()
	config = tf.ConfigProto(log_device_placement=True)
	# config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	# Load the pretrained model
	tim = 0
	print('Loading weights from the pre-trained model')
	weight_initiallizer.restore(sess, '../experiment_SRResnet_scalfin/model-28000')


#----------------------------------------------------------------------------------

fd1 = sys.argv[3]
fd = sys.argv[1].strip('[]').split(',')
scale = int(sys.argv[2])

for fi in fd:
	fl = glob.glob(fi + "/*.jpg")
	for fil in fl:
		im = sic.imread(fil, mode="RGB").astype(np.float32)
		im = np.array([im]).astype(np.float32)
		im = im/ np.max(im)
		with strategy.scope():
			results = sess.run(save_fetch, feed_dict={inputs_raw: bas, path_LR: "/", scal: scale})
		with open(fd1 + "/" + os.path.basename(fil), "wb") as f:
			f.write(results["outputs"][0])

