from layer import *
from helpers import *
import numpy as np

class ErrorLayer(Layer):

	def __init__( self, type):
		print "* Error: type: "+type
		Layer.__init__(self, 'error')
		self.error_type = type
		self.output_error = 0
		self.input = 0

	def forward(self, pred, ground_true):

		pred_shape = pred.shape
		self.input = pred
		pred = pred.flatten()
		if pred.shape != ground_true.shape:
			raise ValueError("Predictions and Ground_true have different shapes! (" + str(pred.shape) + "; "+ str(ground_true.shape) +")") 
		
		self.output_error = pred - ground_true

		entropy = -np.sum(ground_true * np.log(pred+0.00001))

		#print "gt:", ground_true,"pred:", pred,np.log(pred), "ERR:\n",self.output_error

		#self.output_error = np.reshape(np.absolute(np.subtract(ground_true, pred)),pred_shape)
		#print "OUTE: ", self.output_error, "N:", ny, "N2:",nh
		return entropy

	def backward(self):
		return astype(self.output_error)

	def __del__(self):
		class_name = self.__class__.__name__
