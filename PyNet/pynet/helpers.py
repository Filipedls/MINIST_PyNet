import numpy as np

def initWeights(shape, range):
	return np.random.uniform(low=-range, high=range, size=shape)

def initWeights_xavier(shape):
	range = np.sqrt(2. / np.prod(shape[1:]) ) # Xavier 
	return np.random.uniform(low=-range, high=range, size=shape)

def zeros(shape):
	return np.zeros(shape)