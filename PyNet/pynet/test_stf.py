import numpy as np
#import matplotlib.pyplot as plt
import cv2
from timeit import default_timer as timer
#import pyximport; pyximport.install()
from im2col_cy import *
from im2col import *


def im2col_m(img, k_size, padding, stride):

	input_size = k_size[0]*k_size[1]*img.shape[1]
	output_shape = tuple(np.subtract(np.add(img.shape[2:4],2*padding), k_size)/stride + 1)
	vec_input = np.empty((output_shape[0]*output_shape[1],input_size))

	i = 0
	y_range = range(0,output_shape[1])
	for x_start in range(0,output_shape[0]):
		x_end = x_start+k_size[0]

		for y_start in y_range:
			y_end = y_start+k_size[1]

			vec_input[i,:] = img[0,:,x_start:x_end, y_start:y_end].flatten()
			i += 1

	return vec_input


C = 2
H = 4
W =4

cenas = np.ones((C,W,H))
#for ic in range(C):
#	for ih in range(H):
#		for iw in range(W):
#			cenas[ic, ih, iw] = ic + (ih+iw/10.0)/10.0

#cenas = np.array([cenas, cenas])
cenas = np.expand_dims(cenas, axis=0)

output_shape = tuple(np.subtract(np.add(cenas.shape[2:4],2*0), (3,3) )/1 + 1)

n_tries = 1
time = 0.0
x_cols = np.zeros(( 1 * output_shape[0] * output_shape[1], C * 3 * 3))
for i in range(n_tries):
	start = timer()
	x_cols = im2col_cython(cenas, 3, 3, 1, output_shape[0], output_shape[1])
	time += timer() - start

print " %.4f ms"%(time/n_tries*1000)

#print cenas
print "COLS: \n", x_cols.shape

start = timer()
x_cols = im2col_m(cenas, (3, 3), 0, 1)
print " %.4f ms"%((timer() - start)*1000)

print "COLS2: \n", x_cols


print col2im_cython(x_cols, 1, C, H, W, 3, 3, 0, 1, output_shape[0], output_shape[1])



