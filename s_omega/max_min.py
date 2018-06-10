#!/usr/bin/python

import os,fnmatch
import numpy as np

states = os.listdir("/data1/kranti/pratyush/generative/s_omega_2/data/nn2/")
data_dir = "/data1/kranti/pratyush/generative/s_omega_2/data/nn2/"

pattern = '*_grd_truth.npy'
file_names = []

for entry in states:
	if fnmatch.fnmatch(entry, pattern):
		# print(entry)
		file_names.append(entry)


for file in file_names:
	data = np.load(data_dir + file)
	data = data[3000:,:]

	print("------------------------------------------")
	print('{:50}'.format(file), 'max :', '{:13.9f}'.format(np.amax(data[:,0])), 
		'min :', '{:.9f}'.format(np.amin(data[:,0])))

	print('{:50}'.format(file), 'max :', '{:13.9f}'.format(np.amax(data[:,1])), 
		'min :', '{:.9f}'.format(np.amin(data[:,1])))

	print('{:50}'.format(file), 'max :', '{:13.9f}'.format(np.amax(data[:,2])), 
		'min :', '{:.9f}'.format(np.amin(data[:,2])))

