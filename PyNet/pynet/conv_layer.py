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
		#self.d_input = 0

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
		
		#output = np.zeros(self.output_shape)
		#input_exp = np.expand_dims(input, axis=0)
		vec_input = im2col_cython(input, self.kern_size[1], self.kern_size[2], self.stride, self.output_shape[1], self.output_shape[2])
		#print "vec_input:\n", vec_input[0:10,:], "\ninput:\n", input
		# vec_input = empty((self.output_shape[1]*self.output_shape[2],self.input_size))

		# i = 0
		# for x_start in range(0,self.output_shape[1]):
		# 	x_end = x_start+self.kern_size[1]

		# 	for y_start in range(0,self.output_shape[2]):
		# 		y_end = y_start+self.kern_size[2]

		# 		vec_input[i,:] = input[:,x_start:x_end, y_start:y_end].flatten()
		# 		i += 1

				#for n_filter in range(0, self.n_filters):
				#	weights = self.weights[n_filter,:,:,:] # TODO: What about padding?!

				#	output[n_filter, x_start, y_start] = LeReLU(np.asarray(np.sum(kern_input * weights) + self.bias[n_filter]))

		output_vec = np.dot(vec_input, self.weights) + self.bias

		# Split the output by images, to then reshape it correctly
		output_vec = output_vec.reshape((-1,self.output_shape[1]*self.output_shape[2],self.output_shape[0])).transpose(0,2,1)

		#print "weights:\n", self.weights, "\nout_vec:\n", output_vec.shape


		output = np.reshape(output_vec,(input.shape[0],)+self.output_shape) # check this!!!!

		output = self.act.activate(output)

		self.input_shape = input.shape
		self.input_vec = vec_input
		return output

	def backward(self, d_output_error):

		#d_input = zeros(self.input_shape)

		d_x_output = self.act.diff(d_output_error)#d_x_output = LeReLU_derivative(d_output_error)

		self.d_bias += np.sum(d_x_output, axis=(0, 2, 3))

		d_x_output_vec = d_x_output.transpose(0, 2, 3, 1).reshape(-1,self.n_filters).T

		#print "out vec: \n", d_x_output_vec, "\ninput_vec\n", self.input_vec

		self.d_weights += d_x_output_vec.dot(self.input_vec).T#reshape(self.weights.shape)

		d_input_vec = np.dot(self.weights, d_x_output_vec).T

		#print "d_input_vec:\n", d_input_vec

		d_input = col2im_cython(d_input_vec, self.input_shape[0], self.input_shape[1],self.input_shape[2] , self.input_shape[3], 
			self.kern_size[1], self.kern_size[2], self.stride, self.output_shape[1], self.output_shape[2])

		#d_input = np.squeeze(d_input, axis=0)

		# i = 0
		# for x_start in range(0,d_output_error.shape[1]):
		# 	x_end = x_start+self.kern_size[1]

		# 	for y_start in range(0,d_output_error.shape[2]):
		# 		y_end = y_start+self.kern_size[2]

		# 		#d_x_output_xy = d_x_output[:,x_start,y_start].flatten()

		# 		d_input[:,x_start:x_end, y_start:y_end] += np.reshape(d_input_vec[:,i], self.kern_size)
		# 		i += 1
				#input_xy = self.input[:,x_start:x_end, y_start:y_end].flatten()
				#self.d_weights += np.outer(input_xy, d_x_output_xy)#np.dot(np.transpose(self.input), d_x_output)
				#self.d_bias += d_x_output_xy

				#for n_filter in range(0, self.n_filters):
				#	weights = self.weights[n_filter,:,:,:] 
				#	d_input[:,x_start:x_end, y_start:y_end] += d_x_output[n_filter,x_start,y_start] * weights
				#	self.d_weights[n_filter,:,:,:] += d_x_output[n_filter,x_start,y_start] * self.input[:,x_start:x_end, y_start:y_end]
				#	self.d_bias[n_filter] = d_x_output[n_filter,x_start,y_start].flatten()


		#print "D IP SHAPE: " + str(d_input.shape)
		#self.d_input = d_input
		if self.padding > 0:
			return d_input[:, :, self.padding:-self.padding, self.padding:-self.padding]
		return d_input

	def __del__(self):
		class_name = self.__class__.__name__
