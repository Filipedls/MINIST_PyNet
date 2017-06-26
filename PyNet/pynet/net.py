from conv_layer import *
from error_layer import *
from fc_layer import *
from max_pool_layer import *
import random
import cv2
from timeit import default_timer as timer

class Net:

	def __init__( self, layers_def):
		print "=> NET:"
		self.layers = []
		for layer_def in layers_def:
			if layer_def[0] == 'conv':
				layer = ConvLayer(input_size, layer_def[2], (layer_def[1], layer_def[1]), layer_def[3], layer_def[4])
				input_size = layer.output_shape
			elif layer_def[0] == 'fc':
				layer = FCLayer(input_size, layer_def[1], layer_def[2])
				input_size = (layer.n_neurons,)
			elif layer_def[0] == 'maxpool':
				layer = MaxPoolLayer(input_size, (layer_def[1], layer_def[1]), layer_def[2])
				input_size = layer.output_shape
			elif layer_def[0] == 'input':
				input_size = (layer_def[1], layer_def[2], layer_def[3])
				self.input_size = input_size
				print "* Input -> Size: ", input_size
				continue
			elif layer_def[0] == 'error':
				layer = ErrorLayer(layer_def[1])
			else:
				raise ValueError("*** UNKOWN LAYER TYPE ("+layer_def[0]+")")
			self.layers.append(layer)

		self.n_classes = input_size
		print "* NET: done initializing net!"

	def forward(self, input, ground_true):
		layer_input = input
		#print "F times:"
		for layer in self.layers:
			start = timer()
			if layer.type == 'error':
				layer_input = layer.forward(layer_input, ground_true)
			else:
				layer_input = layer.forward(layer_input)
			#print layer.type+" %.2f ms"%((timer() - start)*1000)
		return layer_input

	def backward(self):
		#print "B times:"
		for layer in reversed(self.layers):
			start = timer()
			if layer.type == 'error':
				d_output_error = layer.backward()
			else:
				d_output_error = layer.backward(d_output_error)
			#print layer.type+" %.2f ms"%((timer() - start)*1000)

	def update_weights(self, learning_rate):
		for layer in self.layers:
			if layer.type != 'error':
				layer_input = layer.update_weights(learning_rate)
		#print "* NET: done updating weights."
		return True

	def net_checks(self, learning_rate):

		scale = 0
		n = 0
		for layer in self.layers:
			if layer.type != 'error' and layer.type != 'maxpool':
				param_scale = np.linalg.norm(layer.weights.ravel())
				update = -learning_rate * layer.d_weights
				update_scale = np.linalg.norm(update.ravel())
				n += 1
				scale += update_scale / param_scale

		return scale/n

	#def __del__(self):