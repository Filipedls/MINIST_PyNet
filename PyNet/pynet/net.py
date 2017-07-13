from conv_layer import *
from error_layer import *
from fc_layer import *
from max_pool_layer import *
import random
import cv2
from timeit import default_timer as timer
import pickle

class Net:

	def __init__( self, layers_def):
		"""
		Network initialization, trough the layers_def
		"""
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

		self.n_classes = int(input_size[0])
		self.layers_def = layers_def
		print "* NET: done initializing net!"

	def forward(self, input, ground_true):
		"""
		The Forward pass of the net
		"""
		layer_input = input
		for layer in self.layers:
			start = timer()
			if layer.type == 'error':
				layer_input = layer.forward(layer_input, ground_true)
			else:
				layer_input = layer.forward(layer_input)
		return layer_input

	def backward(self):
		"""
		The Backward pass of the net
		"""
		for layer in reversed(self.layers):
			start = timer()
			if layer.type == 'error':
				d_output_error = layer.backward()
			else:
				d_output_error = layer.backward(d_output_error)

	def save_weights(self, file_path):
		"""
		Saves the weights to a pickle file
		"""
		print "* NET: Saving my weights! (to: "+file_path+")"
		weights = []
		for layer in self.layers:
			if layer.has_weights:
				weights.append(layer.weights)

		# Saving the weights
		with open(file_path, 'w') as f:
		    pickle.dump(weights, f)
		    #np.save(f, weights, allow_pickle=False)

	def load_weights(self, file_path):
		"""
		Loads the weights from a pickle file
		(make sure thats the same net, i dont really check that)
		"""
		print "* NET: Loading my weights! (from: "+file_path+")"
		# Getting back the weights
		with open(file_path) as f:
		    weights = pickle.load(f)
		    #weights = np.load(f)

		weights_i = iter(weights)
		l_weight = next(weights_i)
		for layer in self.layers:
			if layer.has_weights:
				if layer.weights.shape == l_weight.shape:
					layer.weights = l_weight
					try:
						l_weight = next(weights_i)
					except StopIteration:
						pass
				else:
					print "* load_weights: shape doesn't match - net_w:", layer.weights.shape, "load_w:",l_weight.shape

	def check_weights(self, n):
		"""
		return the weights of the n layer of the net, while checking 
		its min and max and normalizing it, to visulization
		"""
		layer = self.layers[n]
		#for layer in self.layers:
		if layer.type == 'convolution':
			weights = layer.weights.reshape((layer.n_filters,)+layer.kern_size)
			weights = weights.transpose((1,2,0,3)).reshape(layer.kern_size[0],layer.kern_size[1],layer.kern_size[2]*layer.n_filters)
			w_min = np.min(weights)
			w_max = np.max(weights)
			weights = (weights - w_min)/(w_max-w_min)
			print 'Max:', w_max, 'MIN:', w_min

		return weights

	#def __del__(self):