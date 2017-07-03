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

layer = ConvLayer( input.shape, 1, (3,3), 1, 1)

layer.weights = np.ones(layer.weights.shape)

output = layer.forward(input)
print "IN: \n", input, "\nOUT: \n", output

diff = np.ones(output.shape)

print "\nDIFF: \n", diff, "\nOUT: \n", layer.backward(diff)