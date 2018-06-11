#!/usr/bin/python

import tensorflow as tf
import numpy as np
import math
import argparse

parser = argparse.ArgumentParser(description="Trajectory")
parser.add_argument('file_name', help='file_name')
args = parser.parse_args()

file_name = args.file_name

output_dir 		= "/data1/kranti/pratyush/quadrotor_model/s_omega/output_dir/"
state_dir	 	= "/data1/kranti/pratyush/quadrotor_model/data/states/"
control_dir 	= "/data1/kranti/pratyush/quadrotor_model/data/controls/"
alpha_dir 		= "/data1/kranti/pratyush/quadrotor_model/data/nn2/"

state   = np.load(state_dir 	+ 's_'+file_name+'.npy')
control = np.load(control_dir   + 'u_'+file_name+'.npy')
alpha   = np.load(alpha_dir     + 'nn2_'+file_name+'_grd_truth.npy')


state = np.concatenate((state[:,0:6], state[:,12:18]),axis=1)

Xtr    = np.concatenate((state[:,3:12], control[:,0:4]), axis = 1 ) #controls
Xtr    = Xtr[:-1,:] # decrease_state 

Ytr    = alpha[1:,0:3]
# Ytr = np.concatenate((Ytr[:,0:2]  , Ytr[:,2:3]),axis=1)



##########################
# Exponential Average
##########################

min_val_X = np.load('min_val_X.npy')
max_val_X = np.load('max_val_X.npy')

min_val_Y = np.load('min_val_Y.npy')
max_val_Y = np.load('max_val_Y.npy')

for col in range(0,Xtr.shape[1]):
	Xtr[:,col] = (( Xtr[:,col] - min_val_X[col] ) / ( max_val_X[col] - min_val_X[col] ))

for col in range(0,Ytr.shape[1]):
	Ytr[:,col] = (( Ytr[:,col] - min_val_Y[col] ) / ( max_val_Y[col] - min_val_Y[col] ))


##########################
# Normalization
###########################

# data_mean_X = np.load('data_mean_X.npy')
# data_std_X = np.load('data_std_X.npy')

# data_mean_Y = np.load('data_mean_Y.npy')
# data_std_Y = np.load('data_std_Y.npy')

# for col in range(0, Xtr.shape[1]):
# 	Xtr[:,col] = ( Xtr[:,col] - data_mean_X[col] ) / ( data_std_X[col] )

# for col in range(0, Ytr.shape[1]):
# 	Ytr[:,col] = ( Ytr[:,col] - data_mean_Y[col] ) / ( data_std_Y[col] )


#######################
# Final Data
#######################

Xtr = Xtr 
Ytr = Ytr 

print(Xtr.shape)
print(Ytr.shape)


# exit()



######################
## Network Structure
#######################

size = Xtr.shape[0]
NUM_SAMPLES = Xtr.shape[0]

INPUT_LAYER =  100
OUTPUT_LAYER = 100
LAYER_1 = 100
LAYER_2 = 100


INPUT_DIM = 13
OUTPUT_DIM = 3


X = tf.placeholder(tf.float32, shape=[None, INPUT_DIM])
Y = tf.placeholder(tf.float32, shape=[None, OUTPUT_DIM])

W = tf.Variable(tf.truncated_normal([INPUT_DIM,INPUT_LAYER]), name="W")
B = tf.Variable(tf.zeros([INPUT_LAYER]), name="B")

W1 = tf.Variable(tf.truncated_normal([INPUT_LAYER, LAYER_1]), name="W1")
B1 = tf.Variable(tf.zeros([LAYER_1]), name="B1")

# W2 = tf.Variable(tf.truncated_normal([LAYER_1, LAYER_2]), name="W2")
# B2 = tf.Variable(tf.zeros([LAYER_2]), name="B2")

# W3 = tf.Variable(tf.truncated_normal([LAYER_2, LAYER_3]), name="W3")
# B3 = tf.Variable(tf.zeros([LAYER_3]), name="B3")

# W4 = tf.Variable(tf.truncated_normal([LAYER_3, LAYER_4]), name="W4")
# B4 = tf.Variable(tf.zeros([LAYER_4]), name="B4")

# W5 = tf.Variable(tf.truncated_normal([LAYER_4, LAYER_5]), name="W5")
# B5 = tf.Variable(tf.zeros([LAYER_5]), name="B5")

# W6 = tf.Variable(tf.truncated_normal([LAYER_5, OUTPUT_LAYER]), name="W6")
# B6 = tf.Variable(tf.zeros([OUTPUT_LAYER]), name="B6")

w = tf.Variable(tf.truncated_normal([OUTPUT_LAYER,OUTPUT_DIM]), name="w")
b = tf.Variable(tf.zeros([OUTPUT_DIM]), name="b")


# Hidden layer 1
h1 = tf.nn.tanh(tf.add(tf.matmul(X,W),B))
# o1 = tf.nn.dropout(h1,0.5)


# output layer
pred_y = tf.matmul(h1,w)

# cost = tf.reduce_sum(tf.pow(pred_y - Y, 2))/(2*BATCH_SIZE)
cost = tf.reduce_sum(tf.pow(pred_y - Y, 2))



saver = tf.train.Saver()

# optimizer = tf.train.AdamOptimizer().minimize(cost)
init = tf.global_variables_initializer()

with tf.Session() as sess:
	saver.restore(sess,output_dir+"models/model.ckpt")
	print("Model Loaded Successfully")

	predictions, c = sess.run([pred_y,cost],feed_dict={X: Xtr ,Y: Ytr})
	print("Cost = ", '{:.5f}'.format(c))
	print("shape = ",predictions.shape)
	np.save('predictions/'+file_name+'_prediction.npy',predictions)