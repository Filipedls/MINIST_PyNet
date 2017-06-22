import numpy as np


b = np.array([np.ones(( 2, 2)),np.ones((2, 2))*2,np.ones(( 2, 2))*3,np.ones(( 2, 2))*4])

print b

print "\nT:",b.transpose(1, 2, 0)

print "\nALL:",b.transpose(1, 2, 0).reshape(4, -1)