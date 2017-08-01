from context import pynet
import numpy as np
import cv2
import math
import random


layer = pynet.MaxPoolLayer((2,4,4), (2,2), 2)

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

input = np.expand_dims(input, axis=0)
input = pynet.astype(input)


print "IN: \n", input, "\nOUT: \n", layer.forward(input)

diff = np.ones((2,2,2))
diff = np.expand_dims(diff, axis=0)

print "\nDIFF: \n", diff, "\nOUT: \n", layer.backward(diff)