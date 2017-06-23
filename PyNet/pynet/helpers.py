import numpy as np

my_dtype = np.float64

def initWeights(shape, range):
	return np.random.uniform(low=-range, high=range, size=shape, dtype = my_dtype)

def initWeights_xavier(shape):
	range = np.sqrt(2. / np.prod(shape[1:]) ) # Xavier 
	return np.random.uniform(low=-range, high=range, size=shape).astype(my_dtype)

def zeros(shape):
	return np.zeros(shape, dtype = my_dtype)

def empty(shape):
	return np.empty(shape, dtype = my_dtype)

def astype(narray):
	return narray.astype(my_dtype)