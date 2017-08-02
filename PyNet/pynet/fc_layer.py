from layer import *
from helpers import *
from activation import *
import numpy as np

class FCLayer(Layer):

	def __init__( self, input_shape, n_neurons, act_type):

		Layer.__init__(self, 'fc', True)

		self.input_shape = input_shape
		self.input_size = np.prod(np.array(input_shape))
		if len(input_shape) > 1:
			self.do_reshape = True
		else:
			self.do_reshape = False
			
		self.n_neurons = n_neurons
		self.bias = zeros(self.n_neurons)#initWeights(n_neurons, 0.01)
		self.weights = initWeights_xavier((self.input_size,self.n_neurons))#initWeights((self.input_size, self.n_neurons), 0.01)#
		self.input = 0
		self.d_weights = zeros(self.weights.shape)
		self.d_bias = zeros(self.bias.shape)

		self.act = Activation(act_type)

		print "* FC -> Input: %d; neurons: %d; do_reshape: " % (self.input_size, self.n_neurons), self.do_reshape

	# TODO: optimize
	def forward(self, input):
		# Reshaping the input, when its not a vector
		if self.do_reshape:
			input = input.reshape(input.shape[0], self.input_size)

		if input.shape[1] != self.input_size:
			raise ValueError("WRONG INPUT SHAPE: " + str(input.shape) + "; weights_shape: "+str(self.input_size))

		# The multiplication of the input by the weights
		z = np.dot(input, self.weights) + self.bias

		# Activation of the output
		output = self.act.activate(z)

		#for n in range(0, self.n_neurons):
		#	weights = self.weights[n,:]
		#
		#	output[n] = LeReLU(np.asarray(np.sum(input * weights) + self.bias[n]))

		self.input = input
		return output

	def backward(self, d_output_error):
		# The derivative of the error of the output
		d_x_output = self.act.diff(d_output_error)

		# The derivative of the input (output of the previous layer)
		d_input = np.dot(d_x_output, self.weights.T)
		
		# The derivative of the weights with respect to the error 
		self.d_weights += np.dot(self.input.T, d_x_output)

		# The derivative of the bias with respect to the error 
		self.d_bias += d_x_output.sum(axis=0)

		#for n in range(0, self.n_neurons):
		#	weights = self.weights[n,:] 
		#	d_input += d_x_output[n] * weights
		#	self.d_weights[n,:] += d_x_output[n] * self.input
		#	self.d_bias[n] = d_x_output[n]

		# Going back to the input's shape
		if self.do_reshape:
			d_input = d_input.reshape((d_output_error.shape[0],)+self.input_shape)

		return d_input

	def __del__(self):
		class_name = self.__class__.__name__
