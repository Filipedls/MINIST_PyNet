from max_pool_layer import *
from helpers import *
import numpy as np
import cv2
import math
import random


layer = MaxPoolLayer((2,4,4), (2,2), 2)

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

input = np.expand_dims(input, axis=0).repeat(2,axis=0)

output = layer.forward(input)
print "IN: \n", input, "\nOUT: \n", output

diff = np.ones(output.shape)

print "\nDIFF: \n", diff, "\nOUT: \n", layer.backward(diff)