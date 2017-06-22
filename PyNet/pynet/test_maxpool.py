from net import *
from helpers import *
import numpy as np
import cv2
import math
import random

from max_pool_layer import *

layer = MaxPoolLayer((2,4,4), (2,2), 2)

input = np.array([[[1, 0, 1, 0], 
				  [0, 0, 0, 0],
				  [1, 0, 1, 0],
				  [0, 0, 0, 0],
				 ],
				 [[0, 0, 0, 0], 
				  [0, 1, 0, 1],
				  [0, 0, 0, 0],
				  [0, 1, 0, 1],
				 ]
				 ])


print "IN: \n", input, "\nOUT: \n", layer.forward(input)

diff = np.ones((2,2,2))

print "\nDIFF: \n", diff, "\nOUT: \n", layer.backward(diff)