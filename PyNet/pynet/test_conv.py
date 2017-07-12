from conv_layer import *
from helpers import *
import numpy as np
import cv2
import math
import random



input = astype(np.array([[[1, 0, 1, 0], 
						  [0, 0, 0, 0],
						  [1, 0, 1, 0],
						  [0, 0, 0, 0],
						 ],
						 [[0, 0, 0, 0], 
						  [0, 1, 0, 1],
						  [0, 0, 0, 0],
						  [0, 1, 0, 1],
						 ]
						 ]))
input = np.expand_dims(input, axis=0).repeat(3,axis=0)

layer = ConvLayer( input.shape[1:], 2, (3,3), 0, 1)

layer.weights[:,0] = np.ones(layer.input_size)
layer.weights[:,1] = np.ones(layer.input_size)*2
#layer.bias = np.ones(layer.bias.shape)

output = layer.forward(input)
print "IN: \n", input, "\nOUT: \n", output


diff = astype(np.array([[[1, 0], 
						  [0, 0],
						 ],
						 [[0, 0], 
						  [0, 1],
						 ]
						 ]))
diff = np.expand_dims(diff, axis=0).repeat(3,axis=0)
#diff = np.ones(output.shape)

print "\nDIFF: \n", diff

d_out = layer.backward(diff)

print "d weights: \n", layer.d_weights, "\nd_bias: \n" , layer.d_bias

print "\nD_OUT: \n", d_out