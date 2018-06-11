import numpy as np
import matplotlib.pyplot as plt
import math
import argparse

parser = argparse.ArgumentParser(description="Plot") 
parser.add_argument('file_name', help='file_name')
args = parser.parse_args()
file_name = args.file_name

data_dir= "/data1/kranti/pratyush/quadrotor_model/data/nn2/"
data_dir= "/data1/kranti/pratyush/quadrotor_model/data/nn2/"
pred_dir = "/data1/kranti/pratyush/quadrotor_model/s_omega/predictions/"


predictions = np.load(pred_dir +file_name+'_prediction.npy')
alpha 	 = np.load(data_dir + 'nn2_'+file_name+'_grd_truth.npy')

size = alpha.shape[0]

acc_meas = alpha[1:,0:3]


grd_truth = acc_meas

#######################
## Exponential Average
#######################

min_val_Y = np.load('min_val_Y.npy')
max_val_Y = np.load('max_val_Y.npy')

# ## Average as ground truth
# print(acc_meas.shape[1])
for col in range(0,acc_meas.shape[1]):
	acc_meas[:,col] = (( acc_meas[:,col] - min_val_Y[col] ) / ( max_val_Y[col] - min_val_Y[col] ))
grd_truth = acc_meas

# Measured value as ground truth
# for col in range(0,predictions.shape[1]):
# 	predictions[:,col] = (( (predictions[:,col]) * ( max_val_Y[col] - min_val_Y[col] ) ) + min_val_Y[col] )
# grd_truth = acc_meas


#####################
## Normalized 
#######################


# data_mean_Y = np.load('data_mean_Y.npy')
# data_std_Y = np.load('data_std_Y.npy')


# ## Measured value as ground truth
# # delta_t = 1
# # for col in range(0,3):
# # 	predictions[:,col] = ( (predictions[:,col]) * data_std_Y[col] ) + data_mean_Y[col]
# # grd_truth = acc_meas


# for col in range(0,acc_meas.shape[1]):
# 	acc_meas[:,col] = ( acc_meas[:,col] - data_mean_Y[col] ) / ( data_std_Y[col] )
# grd_truth = acc_meas



# grd_truth = acc_meas

delta_t = 1


r = predictions.shape[0]


print(predictions.shape)
print(grd_truth.shape)

# exit()

t = [] 

pred_0 = []
gt_0 = []

pred_1 = []
gt_1 = []

pred_2 = []
gt_2 = []

# pred_3 = []
# gt_3 = []
sum = 0.0

# r = 4999

for i in range(0,r):
	t.append(i)

for i in range(0,r):
	pred_0.append((predictions[i,0]))
	gt_0.append(grd_truth[i,0])

	pred_1.append((predictions[i,1]/(delta_t)))
	gt_1.append(grd_truth[i,1])

	pred_2.append((predictions[i,2]/(delta_t)))
	gt_2.append(grd_truth[i,2])

	
fig,ax = plt.subplots(nrows=3,ncols=1)

# ax[0].set_ylim(-1,1)
ax[0].plot(pred_0[0:],'r',label='predictions')
ax[0].plot(gt_0[0:],'b',label='grnd_truth')


# ax[1].set_ylim(-0.6,0.6)
ax[1].plot(pred_1[0:],'r',label='predictions')
ax[1].plot(gt_1[0:],'b',label='grnd_truth')


# # ax[2].set_ylim(-0.6,0.6)
ax[2].plot(t,pred_2,'r',label='predictions')
ax[2].plot(t,gt_2,'b',label='grnd_truth')



plt.show()