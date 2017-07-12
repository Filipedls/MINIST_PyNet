import numpy as np
from helpers import *

class Activation:

	LeRelU_slope = astype(np.array(0.01))

	def __init__( self, type):

		if type == 'relu':
			self.diff_mat = 0
			self.activate = self.ReLU
			self.diff = self.ReLU_derivative

		elif type == 'lerelu':
			self.diff_mat = 0
			self.activate = self.LeReLU
			self.diff = self.LeReLU_derivative

		elif type == 'sigmoid':
			self.activate = self.sigmoid
			self.diff = self.sigmoid_derivative

		elif type == 'softmax':
			self.activate = self.softmax
			self.diff = self.softmax_derivative

		self.type = type


	def ReLU( self, x):
		out = x
		self.diff_mat = out < 0
		out[self.diff_mat] = 0.0
		return out

	def ReLU_derivative( self, d_out):
		d_x = d_out
		d_x[self.diff_mat] = 0.0
		return d_x

	def LeReLU( self, x):
		out = x
		self.diff_mat = out < 0
		out[self.diff_mat] = self.LeRelU_slope*x[self.diff_mat]
		return out

	def LeReLU_derivative( self, d_x):
		d_in = astype(d_x)
		#print "slope: ",self.LeRelU_slope, "DIN_: ", d_in
		d_in[self.diff_mat] = self.LeRelU_slope * d_in[self.diff_mat]
		return d_in

	def sigmoid( self, x):
		return 1.0 / (1.0 + exp(-x))

	def sigmoid_derivative( self, d_x):
		return x * (1.0 - d_x)

	def softmax( self, x):
		x = x - np.max(x, axis=1).reshape((-1,1))
		x = np.exp(x)
		x = x/np.sum(x, axis=1).reshape((-1,1))
		return x

	def softmax_derivative( self, d_x):
		return d_x
