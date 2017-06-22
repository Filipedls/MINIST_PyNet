from layer import *
from helpers import *
from activation import *
import numpy as np

class ConvLayer(Layer):

	def __init__( self, input_shape, n_filters, kern_size, padding, stride, act_type='relu'):
		
		Layer.__init__(self, 'convolution')
		self.kern_size = (input_shape[0],)+kern_size
		self.input_size = np.prod(self.kern_size)
		self.n_filters = n_filters
		self.padding = padding
		self.stride = stride
		self.bias = np.zeros(self.n_filters) #initWeights(self.n_filters, 0.01)
		self.weights = initWeights_xavier((self.input_size, self.n_filters))
		self.input = 0
		self.d_weights = np.zeros(self.weights.shape)
		self.d_bias = np.zeros(self.bias.shape)
		self.d_input = 0

		self.output_shape = (self.n_filters,)+tuple(np.subtract(np.add(input_shape[1:3],2*self.padding), self.kern_size[1:3])/self.stride + 1)

		self.act = Activation(act_type)

		print "* Conv -> output_shape: " + str(self.output_shape) + "; K_size: "+str(kern_size)+"; n_filters: %d; padding: %d; stride: %d" % (n_filters, padding, stride)

	# TODO: optimize
	def forward(self, input):
		if input.shape[0] != self.kern_size[0]:
			raise ValueError("WRONG INPUT SHAPE: " + str(input.shape) + "; Kern_shape: "+str(self.kern_size))

		if self.padding > 0:
			npad = ((0, 0), (self.padding, self.padding), (self.padding, self.padding))
			input = np.pad(input, pad_width=npad, mode='constant', constant_values=0)
		
		#output = np.zeros(self.output_shape)

		vec_input = np.zeros((self.output_shape[1]*self.output_shape[2],self.input_size))

		i = 0
		for x_start in range(0,self.output_shape[1]):
			x_end = x_start+self.kern_size[1]

			for y_start in range(0,self.output_shape[2]):
				y_end = y_start+self.kern_size[2]

				vec_input[i,:] = input[:,x_start:x_end, y_start:y_end].flatten()
				i += 1

				#for n_filter in range(0, self.n_filters):
				#	weights = self.weights[n_filter,:,:,:] # TODO: What about padding?!

				#	output[n_filter, x_start, y_start] = LeReLU(np.asarray(np.sum(kern_input * weights) + self.bias[n_filter]))

		output = np.dot(vec_input, self.weights) + self.bias

		output = np.reshape(output.transpose(),self.output_shape)

		output = self.act.activate(output)

		self.input = input
		return output

	def backward(self, d_output_error):

		d_input = np.zeros(self.input.shape)

		d_x_output = self.act.diff(d_output_error)#d_x_output = LeReLU_derivative(d_output_error)

		for x_start in range(0,d_output_error.shape[1]):
			x_end = x_start+self.kern_size[1]

			for y_start in range(0,d_output_error.shape[2]):
				y_end = y_start+self.kern_size[2]

				d_x_output_xy = d_x_output[:,x_start,y_start].flatten()

				d_input[:,x_start:x_end, y_start:y_end] += np.reshape(np.dot(d_x_output_xy, np.transpose(self.weights)), self.kern_size)
			
				input_xy = self.input[:,x_start:x_end, y_start:y_end].flatten()
				self.d_weights += np.outer(input_xy, d_x_output_xy)#np.dot(np.transpose(self.input), d_x_output)
				self.d_bias += d_x_output_xy

				#for n_filter in range(0, self.n_filters):
				#	weights = self.weights[n_filter,:,:,:] 
				#	d_input[:,x_start:x_end, y_start:y_end] += d_x_output[n_filter,x_start,y_start] * weights
				#	self.d_weights[n_filter,:,:,:] += d_x_output[n_filter,x_start,y_start] * self.input[:,x_start:x_end, y_start:y_end]
				#	self.d_bias[n_filter] = d_x_output[n_filter,x_start,y_start].flatten()


		#print "D IP SHAPE: " + str(d_input.shape)
		self.d_input = d_input
		return d_input

	def update_weights(self, learning_rate):
		self.weights -= self.d_weights * learning_rate
		self.bias -= self.d_bias * learning_rate
		self.d_weights = np.zeros(self.weights.shape)
		self.d_bias = np.zeros(self.bias.shape)
		#self.weights[self.weights > 1] = 1
		#self.weights[self.weights < -1] = -1
		return True

	def __del__(self):
		class_name = self.__class__.__name__
