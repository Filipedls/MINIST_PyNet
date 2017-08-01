import numpy as np
#import matplotlib.pyplot as plt
import cv2

C = 2
H = 4
W =4 

cenas = np.ones((C,W,H))
for ic in range(C):
	for ih in range(H):
		for iw in range(W):
			cenas[ic, ih, iw] = ic + (ih+iw/10.0)/10.0

pool_height = 2
pool_width = 2

print cenas

cenas_S= cenas.reshape( C, H / pool_height, pool_height,
                         W / pool_width, pool_width)


print  cenas_S.max(axis=2).max(axis=3)

cenas_S = cenas_S.transpose((0,1,3,2,4))

cenas_col = cenas_S.reshape((-1,4))

print cenas_S, cenas_S.reshape((-1,4))

print cenas_col.argmax(axis=1)

args = cenas_col.argmax(axis=1)

print cenas_col[range(8),args].reshape(2,2,2)


dif = np.zeros(cenas_col.shape)

dif[range(8),args] = np.ones(8)

print dif.reshape(2,2,2,2,2).transpose((0,1,3,2,4)).reshape((C,W,H))
