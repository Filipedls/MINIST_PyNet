from layer import *
from helpers import *
from activation import *
from im2col_cy import *
import numpy as np

class ConvLayer(Layer):

	def __init__( self, input_shape, n_filters, kern_size, padding, stride, act_type='lerelu'):
		
		Layer.__init__(self, 'convolution', True)
		self.kern_size = (input_shape[0],)+kern_size
		self.input_size = np.prod(self.kern_size)
		self.n_filters = n_filters
		self.padding = padding
		self.stride = stride
		self.bias = zeros(self.n_filters) #initWeights(self.n_filters, 0.01)
		self.weights = initWeights_xavier((self.input_size, self.n_filters))#initWeights((self.input_size, self.n_filters), 0.01)#
		self.input = 0
		self.d_weights = zeros(self.weights.shape)
		self.d_bias = zeros(self.bias.shape)

		self.output_shape = (self.n_filters,)+tuple(np.subtract(np.add(input_shape[1:3],2*self.padding), self.kern_size[1:3])/self.stride + 1)

		self.act = Activation(act_type)

		print "* Conv -> output_shape: " + str(self.output_shape) + "; K_size: "+str(kern_size)+"; n_filters: %d; padding: %d; stride: %d" % (n_filters, padding, stride)

	# TODO: optimize
	def forward(self, input):
		if input.shape[1] != self.kern_size[0]:
			raise ValueError("WRONG INPUT SHAPE: " + str(input.shape) + "; Kern_shape: "+str(self.kern_size))

		if self.padding > 0:
			npad = ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding))
			input = np.pad(input, pad_width=npad, mode='constant', constant_values=0)
		
		# The convolution of the input, each convolution input kernel is turned in a vector
		vec_input = im2col_cython(input, self.kern_size[1], self.kern_size[2], self.stride, self.output_shape[1], self.output_shape[2])
		
		# The actual multiplication of the input by the weights
		output_vec = np.dot(vec_input, self.weights) + self.bias

		# Split the output by images, to then reshape it correctly
		output_vec = output_vec.reshape((-1,self.output_shape[1]*self.output_shape[2],self.output_shape[0])).transpose(0,2,1)

		# The reshape of the output to a 3D feature map
		output = np.reshape(output_vec,(input.shape[0],)+self.output_shape)

		output = self.act.activate(output) # TODO: should be done after the maxpool...

		# Saving stuff for the backward step
		self.input_shape = input.shape
		self.input_vec = vec_input
		return output

	def backward(self, d_output_error):

		# The derivative of the error of the output
		d_x_output = self.act.diff(d_output_error)

		# The derivative of the bias with respect to the error 
		self.d_bias += np.sum(d_x_output, axis=(0, 2, 3))

		# Reshaping the derivative of the output to have all the outputs (of all the neurons)
		# for each input kernel in a row
		d_x_output_vec = d_x_output.transpose(0, 2, 3, 1).reshape(-1,self.n_filters).T

		# The derivative of the weights with respect to the error 
		self.d_weights += d_x_output_vec.dot(self.input_vec).T

		# The derivative of the input (output of the previous layer) with respect to the error 
		d_input_vec = np.dot(self.weights, d_x_output_vec).T

		# Going from vectorized input to a 3D derivative of th error (to sum all the error of the 3D input)
		d_input = col2im_cython(d_input_vec, self.input_shape[0], self.input_shape[1],self.input_shape[2] , self.input_shape[3], 
			self.kern_size[1], self.kern_size[2], self.stride, self.output_shape[1], self.output_shape[2])

		# "Unpadding" the error of the input
		if self.padding > 0:
			return d_input[:, :, self.padding:-self.padding, self.padding:-self.padding]
		return d_input

	def __del__(self):
		class_name = self.__class__.__name__
