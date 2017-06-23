import numpy as np
import matplotlib.pyplot as plt
import cv2

cenas = np.empty((2,2), dtype=np.uint8)


print cenas.dtype 
cenas[:,:] = np.ones((2,2))

print cenas.dtype 