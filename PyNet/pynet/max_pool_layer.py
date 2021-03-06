from layer import *
from helpers import *
from im2col_cy import *
import numpy as np

class MaxPoolLayer(Layer):

	def __init__( self, input_shape, kern_size, stride):
		
		Layer.__init__(self, 'maxpool', False)
		self.kern_size = kern_size
		self.stride = stride
		self.input_shape = input_shape

		self.output_shape = (input_shape[0],)+tuple(np.subtract(input_shape[1:3], self.kern_size[1:3])/self.stride + 1)

		self.vec_input_argmax = 0
		self.vec_input_shape = 0

		print "* MaxPool -> output_shape: " + str(self.output_shape) + "; K_size: "+str(kern_size)+"; stride: %d" % (stride)

	# TODO: optimize for when stride == kern_size
	def forward(self, input):
		N = input.shape[0]

		# Shifts the channels (axis=1) to another dimesion (axis=0) so that we can use im2col, once we want the max per channels
		input = input.reshape( (N*input.shape[1],1)+input.shape[2:4] )

		vec_input = im2col_cython(input, self.kern_size[0], self.kern_size[1], self.stride, self.output_shape[1], self.output_shape[2])

		# Saves the position for the backward step
 		self.vec_input_argmax = np.argmax(vec_input,axis=1)

		output = vec_input[range(vec_input.shape[0]),self.vec_input_argmax].reshape( (N,)+self.output_shape)

		# for x_out in range(0,self.output_shape[1])from im2col_cy import *:
		# 	x_start = x_out*self.stride
		# 	x_end = x_start+self.kern_size[0]

		# 	for y_out in range(0,self.output_shape[2]):
		# 		y_start = y_out * self.stride
		# 		y_end = y_start+self.kern_size[1]

		# 		kern_input = input[:,x_start:x_end, y_start:y_end]

		# 		for n_filter in range(0, self.output_shape[0]):

		# 			kern_input = input[n_filter,x_start:x_end, y_start:y_end].flatten()
		# 			input_argmax = kern_input.argmax()
		# 			self.max_pos[n_filter, x_out, y_out] = input_argmax
		# 			output[n_filter, x_out, y_out] = kern_input[input_argmax]

		# Saving stuff for the backward step
		self.vec_input_shape = vec_input.shape

		return output

	def backward(self, d_output_error):

		# N of inputs (batch)
		N = d_output_error.shape[0]

		d_input = zeros(self.vec_input_shape)

		# Propagation of the error to the position where the max of input was
		d_input[range(d_input.shape[0]),self.vec_input_argmax] = d_output_error.flatten()

		d_input = col2im_cython(d_input, N*self.input_shape[0],1,self.input_shape[1] , self.input_shape[2], 
			self.kern_size[0], self.kern_size[1], self.stride, self.output_shape[1], self.output_shape[2])

		# Puting the channels back to the 1st axis
		d_input = d_input.reshape((N,)+self.input_shape)
		
		# for x_start in range(0,d_output_error.shape[1]):
		# 	x_inp = x_start * self.stride

		# 	for y_start in range(0,d_output_error.shape[2]):
		# 		y_inp = y_start * self.stride

		# 		for n_filter in range(0, d_output_error.shape[0]):
		# 			idx = self.max_pos[n_filter, x_start, y_start]
		# 			x = int(x_inp + idx/self.kern_size[0])
		# 			y = int(y_inp + idx%self.kern_size[0])
		# 			d_input[n_filter,x, y] = d_output_error[n_filter,x_start,y_start]

		return d_input

	def update_weights(self, learning_rate):
		return True

	def __del__(self):
		class_name = self.__class__.__name__
