#!/usr/bin/python

import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt

output_dir 		= "/data1/kranti/pratyush/generative/s_omega_2/output_dir/"
s_dir 			= "/data1/kranti/pratyush/generative/s_omega_2/data_high_rates/states/"
control_dir 	= "/data1/kranti/pratyush/generative/s_omega_2/data_high_rates/controls/"
alpha_dir 		= "/data1/kranti/pratyush/generative/s_omega_2/data_high_rates/nn2/"


s_yaw    = np.load(s_dir 		+ 's_yaw.npy')
u_yaw    = np.load(control_dir 	+ 'u_yaw.npy')


#12
s_circle    = np.load(s_dir 		+ 's_circle.npy')
u_circle    = np.load(control_dir 	+ 'u_circle.npy')

#13
s_x_cos_y    = np.load(s_dir 		+ 's_x_cos_y.npy')
u_x_cos_y    = np.load(control_dir 	+ 'u_x_cos_y.npy')

#14
s_x_sin_y    = np.load(s_dir 		+ 's_x_sin_y.npy')
u_x_sin_y    = np.load(control_dir 	+ 'u_x_sin_y.npy')

#15
s_x_cos_z    = np.load(s_dir 		+ 's_x_cos_z.npy')
u_x_cos_z    = np.load(control_dir 	+ 'u_x_cos_z.npy')

#16
s_x_sin_z    = np.load(s_dir 		+ 's_x_sin_z.npy')
u_x_sin_z    = np.load(control_dir 	+ 'u_x_sin_z.npy')

#17
s_x_cos_sin_y    = np.load(s_dir 		+ 's_x_cos_sin_y.npy')
u_x_cos_sin_y    = np.load(control_dir 	+ 'u_x_cos_sin_y.npy')

#18
s_sin_x_y    = np.load(s_dir 		+ 's_sin_x_y.npy')
u_sin_x_y    = np.load(control_dir 	+ 'u_sin_x_y.npy')

#19
s_cos_x_y    = np.load(s_dir 		+ 's_cos_x_y.npy')
u_cos_x_y    = np.load(control_dir  + 'u_cos_x_y.npy')

#20
s_cos_sin_x_y    = np.load(s_dir 		+ 's_cos_sin_x_y.npy')
u_cos_sin_x_y    = np.load(control_dir 	+ 'u_cos_sin_x_y.npy')

#21
s_y_cos_z    = np.load(s_dir 		+ 's_y_cos_z.npy')
u_y_cos_z 	 = np.load(control_dir  + 'u_y_cos_z.npy') 	

#22
s_y_sin_z    = np.load(s_dir 		+ 's_y_sin_z.npy')
u_y_sin_z    = np.load(control_dir  + 'u_y_sin_z.npy')

#23
s_x_rand_y = np.load(s_dir 		 + 's_x_rand_y.npy')
u_x_rand_y = np.load(control_dir + 'u_x_rand_y.npy')

#24
s_rand_x_y = np.load(s_dir		 + 's_rand_x_y.npy')
u_rand_x_y = np.load(control_dir + 'u_rand_x_y.npy')

#25
s_rand_neg_x_y = np.load(s_dir 			+ 's_rand_neg_x_y.npy')
u_rand_neg_x_y = np.load(control_dir    + 'u_rand_neg_x_y.npy')

#26
s_rand_x_neg_y = np.load(s_dir 			+ 's_rand_x_neg_y.npy')
u_rand_x_neg_y = np.load(control_dir    + 'u_rand_x_neg_y.npy')

s_test_traj_1 = np.load(s_dir 			+ 's_test_traj_1.npy')
u_test_traj_1 = np.load(control_dir     + 'u_test_traj_1.npy')





alpha_yaw 	= np.load(alpha_dir + 'nn2_yaw_grd_truth.npy')

alpha_x_cos_y 	  = np.load(alpha_dir + 'nn2_x_cos_y_grd_truth.npy')
alpha_x_sin_y 	  = np.load(alpha_dir + 'nn2_x_sin_y_grd_truth.npy')
alpha_x_cos_z 	  = np.load(alpha_dir + 'nn2_x_cos_z_grd_truth.npy')
alpha_x_sin_z 	  = np.load(alpha_dir + 'nn2_x_sin_z_grd_truth.npy')
alpha_x_cos_sin_y = np.load(alpha_dir + 'nn2_x_cos_sin_y_grd_truth.npy')


alpha_sin_x_y 	  = np.load(alpha_dir + 'nn2_sin_x_y_grd_truth.npy')
alpha_cos_x_y 	  = np.load(alpha_dir + 'nn2_cos_x_y_grd_truth.npy')
alpha_y_cos_z 	  = np.load(alpha_dir + 'nn2_y_cos_z_grd_truth.npy')
alpha_y_sin_z 	  = np.load(alpha_dir + 'nn2_y_sin_z_grd_truth.npy')
alpha_cos_sin_x_y = np.load(alpha_dir + 'nn2_cos_sin_x_y_grd_truth.npy')

alpha_rand_x_y = np.load(alpha_dir + 'nn2_rand_x_y_grd_truth.npy')
alpha_x_rand_y = np.load(alpha_dir + 'nn2_x_rand_y_grd_truth.npy')
alpha_rand_x_neg_y = np.load(alpha_dir + 'nn2_rand_x_neg_y_grd_truth.npy')
alpha_rand_neg_x_y = np.load(alpha_dir + 'nn2_rand_neg_x_y_grd_truth.npy')
alpha_test_traj_1 = np.load(alpha_dir + 'nn2_test_traj_1_grd_truth.npy')

alpha_circle = np.load(alpha_dir + 'nn2_circle_grd_truth.npy')





s_yaw = np.concatenate((s_yaw[:,0:6], s_yaw[:,12:18]),axis=1)

s_x_cos_y = np.concatenate((s_x_cos_y[:,0:6], s_x_cos_y[:,12:18]),axis=1)
s_x_sin_y = np.concatenate((s_x_sin_y[:,0:6], s_x_sin_y[:,12:18]),axis=1)
s_x_cos_sin_y = np.concatenate((s_x_cos_sin_y[:,0:6], s_x_cos_sin_y[:,12:18]),axis=1)
s_x_sin_z = np.concatenate((s_x_sin_z[:,0:6], s_x_sin_z[:,12:18]),axis=1)
s_x_cos_z = np.concatenate((s_x_cos_z[:,0:6], s_x_cos_z[:,12:18]),axis=1)

s_cos_x_y = np.concatenate((s_cos_x_y[:,0:6], s_cos_x_y[:,12:18]),axis=1)
s_sin_x_y = np.concatenate((s_sin_x_y[:,0:6], s_sin_x_y[:,12:18]),axis=1)
s_cos_sin_x_y = np.concatenate((s_cos_sin_x_y[:,0:6], s_cos_sin_x_y[:,12:18]),axis=1)
s_y_sin_z = np.concatenate((s_y_sin_z[:,0:6], s_y_sin_z[:,12:18]),axis=1)
s_y_cos_z = np.concatenate((s_y_cos_z[:,0:6], s_y_cos_z[:,12:18]),axis=1)

s_x_rand_y = np.concatenate((s_x_rand_y[:,0:6], s_x_rand_y[:,12:18]),axis = 1)
s_rand_x_y = np.concatenate((s_rand_x_y[:,0:6], s_rand_x_y[:,12:18]),axis = 1)

s_rand_x_neg_y = np.concatenate((s_rand_x_neg_y[:,0:6], s_rand_x_neg_y[:,12:18]),axis = 1)
s_rand_neg_x_y = np.concatenate((s_rand_neg_x_y[:,0:6], s_rand_neg_x_y[:,12:18]),axis = 1)
s_test_traj_1  = np.concatenate((s_test_traj_1[:,0:6], s_test_traj_1[:,12:18]),axis = 1)


s_circle = np.concatenate((s_circle[:,0:6], s_circle[:,12:18]),axis = 1)




# u_yaw = np.array((u_yaw[:,0:3]),ndmin=2) #remove_controls

# u_x_cos_y 		= np.array((u_x_cos_y[:,0:3]),ndmin=2) #remove_controls
# u_x_sin_y 		= np.array((u_x_sin_y[:,0:3]),ndmin=2) #remove_controls
# u_x_cos_sin_y 	=  np.array((u_x_cos_sin_y[:,0:3]),ndmin=2) #remove_controls
# u_x_sin_z 		= np.array((u_x_sin_z[:,0:3]),ndmin=2) #remove_controls
# u_x_cos_z 		= np.array((u_x_cos_z[:,0:3]),ndmin=2) #remove_controls

# u_cos_x_y 		= np.array((u_cos_x_y[:,0:3]),ndmin=2) #remove_controls
# u_sin_x_y 		= np.array((u_sin_x_y[:,0:3]),ndmin=2) #remove_controls
# u_cos_sin_x_y 	= np.array((u_cos_sin_x_y[:,0:3]),ndmin=2) #remove_controls
# u_y_sin_z 		= np.array((u_y_sin_z[:,0:3]),ndmin=2) #remove_controls
# u_y_cos_z 		= np.array((u_y_cos_z[:,0:3]),ndmin=2) #remove_controls


# u_x_rand_y 		= np.array((u_x_rand_y[:,0:3]),ndmin=2) #remove_controls
# u_rand_x_y 		= np.array((u_rand_x_y[:,0:3]),ndmin=2) #remove_controls
# u_rand_x_neg_y 	= np.array((u_rand_x_neg_y[:,0:3]),ndmin=2) #remove_controls
# u_rand_neg_x_y 	= np.array((u_rand_neg_x_y[:,0:3]),ndmin=2) #remove_controls
# u_test_traj_1 	= np.array((u_test_traj_1[:,0:3]),ndmin=2) #remove_controls
# u_circle 		= np.array((u_circle[:,0:3]),ndmin=2) #remove_controls





yaw    = np.concatenate((s_yaw[:,3:12], u_yaw), axis = 1 ) #controls
yaw    = yaw[:-1,:] # decrease_state 
yaw_gt = alpha_yaw[1:,0:2] #decrease_acc


x_sin_y    = np.concatenate((s_x_sin_y[:,3:12], u_x_sin_y), axis = 1 ) #controls
x_sin_y    = x_sin_y[:-1,:] # decrease_state
x_sin_y_gt = alpha_x_sin_y[1:,0:2] #decrease_acc

x_cos_y    = np.concatenate((s_x_cos_y[:,3:12], u_x_cos_y), axis = 1 ) #controls
x_cos_y    = x_cos_y[:-1,:] # decrease_state
x_cos_y_gt = alpha_x_cos_y[1:,0:2] #decrease_acc


x_sin_z    = np.concatenate((s_x_sin_z[:,3:12], u_x_sin_z), axis = 1 ) #controls
x_sin_z    = x_sin_z[:-1,:] # decrease_state 
x_sin_z_gt = alpha_x_sin_z[1:,0:2] #decrease_acc


x_cos_z    = np.concatenate((s_x_cos_z[:,3:12], u_x_cos_z), axis = 1 ) #controls
x_cos_z    = x_cos_z[:-1,:] # decrease_state 
x_cos_z_gt = alpha_x_cos_z[1:,0:2] #decrease_acc


x_cos_sin_y    = np.concatenate((s_x_cos_sin_y[:,3:12], u_x_cos_sin_y), axis = 1 ) #controls
x_cos_sin_y    = x_cos_sin_y[:-1,:] # decrease_state 
x_cos_sin_y_gt = alpha_x_cos_sin_y[1:,0:2] #decrease_acc


sin_x_y    = np.concatenate((s_sin_x_y[:,3:12], u_sin_x_y), axis = 1 ) #controls
sin_x_y    = sin_x_y[:-1,:] # decrease_state 
sin_x_y_gt = alpha_sin_x_y[1:,0:2] #decrease_acc


cos_x_y    = np.concatenate((s_cos_x_y[:,3:12], u_cos_x_y), axis = 1 ) #controls
cos_x_y    = cos_x_y[:-1,:] # decrease_state 
cos_x_y_gt = alpha_cos_x_y[1:,0:2] #decrease_acc


cos_sin_x_y    = np.concatenate((s_cos_sin_x_y[:,3:12], u_cos_sin_x_y), axis = 1 ) #controls
cos_sin_x_y    = cos_sin_x_y[:-1,:] # decrease_state 
cos_sin_x_y_gt = alpha_cos_sin_x_y[1:,0:2] #decrease_acc


y_sin_z    = np.concatenate((s_y_sin_z[:,3:12], u_y_sin_z), axis = 1 ) #controls
y_sin_z    = y_sin_z[:-1,:] # decrease_state 
y_sin_z_gt = alpha_y_sin_z[1:,0:2] #decrease_acc

y_cos_z    = np.concatenate((s_y_cos_z[:,3:12], u_y_cos_z), axis = 1 ) #controls
y_cos_z    = y_cos_z[:-1,:] # decrease_state 
y_cos_z_gt = alpha_y_cos_z[1:,0:2] #decrease_acc

x_rand_y    = np.concatenate((s_x_rand_y[:,3:12], u_x_rand_y), axis = 1 ) #controls
x_rand_y    = x_rand_y[:-1,:] # decrease_state 
x_rand_y_gt = alpha_x_rand_y[1:,0:2] #decrease_acc


rand_x_y    = np.concatenate((s_rand_x_y[:,3:12], u_rand_x_y), axis = 1 ) #controls
rand_x_y    = rand_x_y[:-1,:] # decrease_state 
rand_x_y_gt = alpha_rand_x_y[1:,0:2] #decrease_acc


rand_x_neg_y    = np.concatenate((s_rand_x_neg_y[:,3:12], u_rand_x_neg_y), axis = 1 ) #controls
rand_x_neg_y    = rand_x_neg_y[:-1,:] # decrease_state 
rand_x_neg_y_gt = alpha_rand_x_neg_y[1:,0:2] #decrease_acc


rand_neg_x_y    = np.concatenate((s_rand_neg_x_y[:,3:12], u_rand_neg_x_y), axis = 1 ) #controls
rand_neg_x_y    = rand_neg_x_y[:-1,:] # decrease_state 
rand_neg_x_y_gt = alpha_rand_neg_x_y[1:,0:2] #decrease_acc


test_traj_1    = np.concatenate((s_test_traj_1[:,3:12], u_test_traj_1), axis = 1 ) #controls
test_traj_1    = test_traj_1[:-1,:] # decrease_state 
test_traj_1_gt = alpha_test_traj_1[1:,0:2] #decrease_acc


circle    = np.concatenate((s_circle[:,3:12], u_circle), axis = 1 ) #controls
circle    = circle[:-1,:] # decrease_state 
circle_gt = alpha_circle[1:,0:2] #decrease_acc



#############################
# Trajectory
#############################

# train_X = np.concatenate((y,    xy   , x   , x_y   , _y,    _x_y    , _x   , _xy   , z   ,   x_sin_y,    x_sin_z    , x_cos_z,    x_cos_sin_y    , sin_x_y    ,  cos_x_y    ,  y_sin_z   , x_rand_y   , rand_x_y    , rand_x_neg_y    , rand_neg_x_y )) 
# train_Y = np.concatenate((y_gt, xy_gt, x_gt, x_y_gt, _y_gt, _x_y_gt , _x_gt, _xy_gt, z_gt,   x_sin_y_gt, x_sin_z_gt , x_cos_z_gt, x_cos_sin_y_gt , sin_x_y_gt ,  cos_x_y_gt ,  y_sin_z_gt, x_rand_y_gt, rand_x_y_gt , rand_x_neg_y_gt , rand_neg_x_y_gt )) 

# train_X = np.concatenate((y,    xy   , x   , x_y   , _y,    _x_y    , _x   , _xy   , z   ,   x_sin_y,    x_sin_z    , x_cos_z,    x_cos_sin_y    , sin_x_y    ,  cos_x_y    ,  y_sin_z   , x_rand_y[0:5000,:]   , rand_x_y[0:5000,:]    , rand_x_neg_y[0:5000,:]    , rand_neg_x_y[0:5000,:] )) 
# train_Y = np.concatenate((y_gt, xy_gt, x_gt, x_y_gt, _y_gt, _x_y_gt , _x_gt, _xy_gt, z_gt,   x_sin_y_gt, x_sin_z_gt , x_cos_z_gt, x_cos_sin_y_gt , sin_x_y_gt ,  cos_x_y_gt ,  y_sin_z_gt, x_rand_y_gt[0:5000,:], rand_x_y_gt[0:5000,:] , rand_x_neg_y_gt[0:5000,:] , rand_neg_x_y_gt[0:5000,:] )) 


test_X = np.concatenate((circle,     test_traj_1))
test_Y = np.concatenate((circle_gt , test_traj_1_gt))

# train_X = np.concatenate((x_rand_y   , rand_x_y    ,x_rand_sin_y     , rand_sin_x_y))
# train_Y = np.concatenate((x_rand_y_gt, rand_x_y_gt ,x_rand_sin_y_gt , rand_sin_x_y_gt))

# train_X = np.concatenate((x_sin_y,    x_cos_y))
# train_Y = np.concatenate((x_sin_y_gt, x_cos_y_gt))


# train_X = np.concatenate((y[0:10000,:],    xy[0:10000,:]   , x[0:10000,:]   , x_y[0:10000,:]   , _y[0:10000,:],    _x_y[0:10000,:]    , _x[0:10000,:]   , _xy[0:10000,:]   , x_sin_y[0:10000,:],     x_sin_z[0:10000,:], 	cos_sin_x_y[0:10000,:],	    x_cos_sin_y[0:10000,:],    x_cos_z[0:10000,:]    ,  sin_x_y[0:10000,:]    , cos_x_y[0:10000,:]   ,  y_sin_z[0:10000,:]   , y_cos_z[0:10000,:]    , x_rand_y[1000:,:]   ,  rand_x_y[1000:,:]   , rand_neg_x_y[1000:,:]   , rand_x_neg_y[1000:,:]))
# train_Y = np.concatenate((y_gt[0:10000,:], xy_gt[0:10000,:], x_gt[0:10000,:], x_y_gt[0:10000,:], _y_gt[0:10000,:], _x_y_gt[0:10000,:] , _x_gt[0:10000,:], _xy_gt[0:10000,:], x_sin_y_gt[0:10000,:],  x_sin_z_gt[0:10000,:],	cos_sin_x_y_gt[0:10000,:] , x_cos_sin_y_gt[0:10000,:], x_cos_z_gt[0:10000,:] ,  sin_x_y_gt[0:10000,:] , cos_x_y_gt[0:10000,:] , y_sin_z_gt[0:10000,:], y_cos_z_gt[0:10000,:] , x_rand_y_gt[1000:,:] , rand_x_y_gt[1000:,:], rand_neg_x_y_gt[1000:,:], rand_x_neg_y_gt[1000:,:]))

train_X = np.concatenate(( x_sin_y,     x_sin_z, 		cos_sin_x_y,	 x_cos_sin_y, 	 x_cos_z    ,  sin_x_y    , cos_x_y    ,  y_sin_z[1000:,:]   , y_cos_z    , x_rand_y[1000:,:]   ,  rand_x_y[1000:,:]   , rand_neg_x_y[1000:,:]   , rand_x_neg_y[1000:,:]))
train_Y = np.concatenate(( x_sin_y_gt,  x_sin_z_gt,	    cos_sin_x_y_gt , x_cos_sin_y_gt, x_cos_z_gt ,  sin_x_y_gt , cos_x_y_gt ,  y_sin_z_gt[1000:,:], y_cos_z_gt , x_rand_y_gt[1000:,:] , rand_x_y_gt[1000:,:], rand_neg_x_y_gt[1000:,:], rand_x_neg_y_gt[1000:,:]))




# train_X = np.concatenate((x_rand_y,    rand_x_y   , rand_neg_x_y   , rand_x_neg_y))
# train_Y = np.concatenate((x_rand_y_gt, rand_x_y_gt, rand_neg_x_y_gt, rand_x_neg_y_gt))

# train_X = np.concatenate((x  ,  _x,    y,    _y,    z,    line,    yaw,    xy   , x_y,   _x_y   , _xy  ))
# train_Y = np.concatenate((x_gt, _x_gt, y_gt, _y_gt, z_gt, line_gt, yaw_gt, xy_gt, x_y_gt,_x_y_gt, _xy_gt))   



# train_Y = np.concatenate((train_Y[:,0:2]*100, train_Y[:,2:3]*1000),axis=1)
# test_Y = np.concatenate((test_Y[:,0:2]*100  , test_Y[:,2:3]*1000),axis=1)

# print(test_Y.shape)
# exit()

###########################################
### Normalization
###########################################

print("-----------------Max-- Min-----------------")

data_X = np.concatenate((train_X ,test_X))
data_Y = np.concatenate((train_Y, test_Y))


min_val_X = []
max_val_X = []

print('Input')
for col in range(0,data_X.shape[1]):
	print('col : ',col, 'min = ','{:.5f}'.format(np.amin(data_X[:,col])), ' max =', '{:.5f}'.format(np.amax(data_X[:,col])))
	min_val_X.append(np.amin(data_X[:,col]))
	max_val_X.append(np.amax(data_X[:,col]))

	train_X[:,col] = ((train_X[:,col] - np.amin(data_X[:,col]))/( np.amax(data_X[:,col]) -  np.amin(data_X[:,col])))
	test_X[:, col] =  ((test_X[:,col] - np.amin(data_X[:,col]))/( np.amax(data_X[:,col]) -   np.amin(data_X[:,col]))) 

# # # exit()

np.save('min_val_X.npy',min_val_X)
np.save('max_val_X.npy',max_val_X)

min_val_Y = []
max_val_Y = []


print('Output')
for col in range(0,data_Y.shape[1]):
	print('col : ',col, 'min = ','{:.5f}'.format(np.amin(data_Y[:,col])), ' max =', '{:.5f}'.format(np.amax(data_Y[:,col])))
	min_val_Y.append(np.amin(data_Y[:,col]))
	max_val_Y.append(np.amax(data_Y[:,col]))

	train_Y[:, col] = ((train_Y[:,col] -np.amin(data_Y[:,col]))/( np.amax(data_Y[:,col]) -  np.amin(data_Y[:,col]))) 
	test_Y[:, col] =  ((test_Y[:,col] - np.amin(data_Y[:,col]))/( np.amax(data_Y[:,col]) -  np.amin(data_Y[:,col]))) 

np.save('min_val_Y.npy',min_val_Y)
np.save('max_val_Y.npy',max_val_Y)

# exit()

##########################
# Regularization
##########################

# data_X = np.concatenate((train_X, test_X))
# data_Y = np.concatenate((train_Y, test_Y))

# data_mean_X = []
# data_std_X = []

# data_mean_Y = []
# data_std_Y = []

# for col in range(0,train_X.shape[1]):
# 	data_mean_X.append(data_X[:,col].mean())
# 	data_std_X.append(data_X[:,col].std())

# 	train_X[:,col] = (train_X[:,col] - data_X[:,col].mean()) / (data_X[:,col].std())
# 	test_X[:,col]  = (test_X[:,col] - data_X[:,col].mean()) / (data_X[:,col].std())

# for col in range(0, train_Y.shape[1]):
# 	data_mean_Y.append(data_Y[:,col].mean())
# 	data_std_Y.append(data_Y[:,col].std())

# 	train_Y[:,col] = (train_Y[:,col] - data_Y[:,col].mean()) / (data_Y[:,col].std())
# 	test_Y[:,col]  = (test_Y[:,col] - data_Y[:,col].mean()) / (data_Y[:,col].std())


# np.save('data_mean_X.npy',data_mean_X)
# np.save('data_std_X.npy',data_std_X)

# np.save('data_mean_Y.npy',data_mean_Y)
# np.save('data_std_Y.npy',data_std_Y)


# exit()

#####################
# Data analyze
####################
# print("-----------------Mean-----------------")

# data_X = np.concatenate((train_X , test_X ))
# data_Y = np.concatenate((train_Y , test_Y ))


# for col in range(0,data_X.shape[1]):
# 	mean = 0.0
# 	for row in range(0,data_X.shape[0]):
# 		mean += np.fabs(data_X[row,col])

# 	print('col : ',col, 'Mean = :', (mean/data_X.shape[0]))


# print('Output')
# for col in range(0,data_Y.shape[1]):
# 	mean = 0.0
# 	for row in range(0,data_Y.shape[0]):
# 		mean += np.fabs(data_Y[row,col])

# 	print('col : ',col, 'Mean = :', (mean/data_Y.shape[0]))

# exit()




############################
## Network Structure
############################


train_X = train_X 
train_Y = train_Y 

test_X = test_X 
test_Y = test_Y 

randomPermut = np.random.permutation(train_X.shape[0])

train_X =  train_X[randomPermut]
train_Y =  train_Y[randomPermut]

NUM_SAMPLES = train_X.shape[0]
BATCH_SIZE = 1000

#
print(train_X.shape)
print(train_Y.shape)

print(test_X.shape)
print(test_Y.shape)
# exit()

INPUT_LAYER =  100 # layer_nodes 
OUTPUT_LAYER = 100 # layer_nodes 
LAYER_1 = 100 # layer_nodes 
LAYER_2 = 100 # layer_nodes 
LAYER_3 = 100 # layer_nodes 
LAYER_4 = 100 # layer_nodes 
LAYER_5 = 100 # layer_nodes 
NUM_EPOCHS = 5000

INPUT_DIM  = 13
OUTPUT_DIM = 2

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

# h2 = tf.nn.tanh(tf.add(tf.matmul(h1,W1),B1))
# # # o2 = tf.nn.dropout(h2,0.5)

# h3 = tf.nn.tanh(tf.add(tf.matmul(h2,W2),B2))
# # # o3 = tf.nn.dropout(h3,0.5)

# h4 = tf.nn.tanh(tf.add(tf.matmul(h3,W3),B3))

# h5 = tf.nn.tanh(tf.add(tf.matmul(h4,W4),B4))

# h6 = tf.nn.relu(tf.add(tf.matmul(h5,W5),B5))

# h7 = tf.nn.relu(tf.add(tf.matmul(h6,W6),B6))


# output layer
pred_y = tf.matmul(h1,w)

# cost = tf.reduce_sum(tf.pow(pred_y - Y, 2))/(2*BATCH_SIZE)
cost = tf.reduce_sum(tf.pow(pred_y - Y, 2))/(2*BATCH_SIZE)






# optimizer = tf.train.MomentumOptimizer(learning_rate=1e-2, momentum=MOMENTUM).minimize(cost)
# optimizer = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(cost)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-2).minimize(cost)

init = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

num_batches = int(math.ceil(float(train_X.shape[0])/BATCH_SIZE))
test_num_batches = int(math.ceil(float(test_X.shape[0])/BATCH_SIZE))
print('train_num_batches', num_batches)
print('test_num_batches', test_num_batches)
# exit()

plot_cost = []

with tf.Session() as sess:
	# Run the initializer

	sess.run(init)
	# Fit all training data
	for epoch in range(NUM_EPOCHS):

		## Training
		running_cost = 0.0
		steps = range(num_batches)
		for step in steps:
				left = step*BATCH_SIZE
				right = min(left + BATCH_SIZE, train_X.shape[0])
				tx = train_X[left:right]
				ty = train_Y[left:right]
				_, r_c = sess.run([optimizer,cost], feed_dict={X: tx, Y: ty})
				running_cost += r_c


		## Testing
		test_running_cost = 0.0
		test_steps = range(test_num_batches)		
		for t_st in test_steps:
			left = t_st*BATCH_SIZE
			right = min(left + BATCH_SIZE, test_X.shape[0])
			tx = test_X[left:right]
			ty = test_Y[left:right]
			t_c = sess.run(cost, feed_dict={X: tx, Y: ty})
			test_running_cost += t_c

		# if (step+1) % LOG_STEP == 0:
		# 	c = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
			# print("Step:", '%04d' % (step+1), "cost=", "{:.9f}".format(c))

		# for (x, y) in zip(train_X, train_Y):
		# 	sess.run(optimizer, feed_dict={X: x, Y: y})

				# sess.run(optimizer, feed_dict={X: train_X, Y: train_Y})
		# Display logs per epoch step
		# c = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
		# _, c = sess.run([optimizer,cost], feed_dict={X:train_X, Y:train_Y})
		epoch_cost = running_cost/num_batches
		test_epoch_cost = test_running_cost/test_num_batches
		print("    Epoch:", '%04d' % (epoch+1), "train_cost=", "{:.9f}".format(epoch_cost) , " test_cost=", "{:.9f}".format(test_epoch_cost))
		plot_cost.append(test_epoch_cost)

	print("Optimization Finished!")
	training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
	
	# WeightLayer0 = sess.run(W)
	# WeightLayer1 = sess.run(w)
	

	# BiasLayer0 = sess.run(B)
	# BiasLayer1 = sess.run(b)

	# print("Training cost=", training_cost)

	# np.save(output_dir + 'WeightLayer0_input_state.npy',WeightLayer0)
	# np.save(output_dir + 'WeightLayer1_input_state.npy',WeightLayer1)
	

	# np.save(output_dir + 'BiasLayer0_input_state.npy',BiasLayer0)
	# np.save(output_dir + 'BiasLayer1_input_state.npy',BiasLayer1)
	

	# Save the variables to disk.
		# save_path = saver.save(sess, "/home/pratyush/Desktop/serb/dnn_trajectory/models/model.ckpt")
		# save_path = saver.save(sess,"/users/ms/pratyushvarshney/Desktop/dnn_trajectory/models/model.ckpt")
	save_path = saver.save(sess, output_dir + "models/model.ckpt")
	print("Model saved in file: %s" % save_path)
	np.save('test_cost.npy',plot_cost)

	plt.plot(plot_cost[10:],'b')
	plt.show()