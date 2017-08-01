from context import pynet
import numpy as np
import cv2
import math
import random



input = pynet.astype(np.array([[[1, 0], 
						  [0, 0]
						 ],
						 [[0, 0], 
						  [0, 1]
						 ]
						 ]))

input = np.expand_dims(input, axis=0).repeat(3,axis=0)

layer = pynet.FCLayer( input.shape[1:], 2,'lerelu')

layer.weights = np.ones(layer.weights.shape)
layer.weights[:,1] = 2
#layer.bias = np.ones(layer.bias.shape)

output = layer.forward(input)
print "IN: \n", input, "\nOUT: \n", output


diff = pynet.astype(np.array([[1, 0],[1,1],[0,1]]))
#diff = np.expand_dims(diff, axis=0).repeat(3,axis=0)
#diff = np.ones(output.shape)

print "\nDIFF: \n", diff, "\nD_OUT: \n", layer.backward(diff)