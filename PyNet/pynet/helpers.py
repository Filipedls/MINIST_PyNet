import numpy as np

my_dtype = np.float64

#TODO: Ortogonal init

def initWeights(shape, range):
	return np.random.uniform(low=-range, high=range, size=shape)#, dtype = my_dtype)

def initWeights_xavier(shape):
	n_in = shape[0]
	w = np.random.standard_normal(shape) * np.sqrt(2.0/n_in)
	#range = np.sqrt(2. / shape[0] ) # Xavier 
	#return np.random.uniform(low=-range, high=range, size=shape)
	return w.astype(my_dtype)

def zeros(shape):
	return np.zeros(shape, dtype = my_dtype)

def empty(shape):
	return np.empty(shape, dtype = my_dtype)

def astype(narray):
	return narray.astype(my_dtype)