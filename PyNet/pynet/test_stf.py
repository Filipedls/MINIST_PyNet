import numpy as np

a = np.ones((4, 3, 2))

# npad is a tuple of (n_before, n_after) for each dimension
npad = ((0, 0), (1, 1), (1, 1))
b = np.pad(a, pad_width=npad, mode='constant', constant_values=0)

print b


ar = np.array([
				[1.1,1.2,1.3],
				[2.1,2.2,2.3],
				[3.1,3.2,3.3],
				[4.1,4.2,4.3],
			  ]).transpose()

print np.reshape(ar,(3,2,2))

lr = 0.005
for i in range(0,5000):
	lr = 0.9992*lr

print lr