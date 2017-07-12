from layer import *
from helpers import *
import numpy as np

class ErrorLayer(Layer):

	def __init__( self, type):
		print "* Error: type: "+type
		Layer.__init__(self, 'error', False)
		self.error_type = type
		self.output_error = 0
		self.input = 0

	def forward(self, pred, ground_true):

		self.input = pred
		#pred = pred.flatten()
		if pred.shape != ground_true.shape:
			raise ValueError("Predictions and Ground_true have different shapes! (" + str(pred.shape) + "; "+ str(ground_true.shape) +")") 
		
		self.output_error = pred - ground_true

		#print "pred:\n", pred, "\ngt:\n", ground_true, "\nout_e:\n", self.output_error, "\nent:\n", -ground_true * np.log(pred+0.00001)

		#entropy = np.sum(-ground_true * np.log(pred))

		entropy = 0
		for i in range(pred.shape[0]):
			entropy += -np.sum(ground_true[i,:] * np.log(pred[i,:]+0.0000001))


		#print "gt:", ground_true,"pred:", pred,np.log(pred), "ERR:\n",self.output_error

		#self.output_error = np.reshape(np.absolute(np.subtract(ground_true, pred)),pred_shape)
		#print "OUTE: ", self.output_error, "N:", ny, "N2:",nh
		return entropy

	def backward(self):
		return astype(self.output_error)

	def __del__(self):
		class_name = self.__class__.__name__
