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
			# Convolutional layer
			if layer_def['layer'] == 'conv':
				layer = ConvLayer(input_size, layer_def['n'], (layer_def['kern_size'], layer_def['kern_size']), layer_def['pad'], layer_def['stride'])
				input_size = layer.output_shape
			# Fully connected layer
			elif layer_def['layer'] == 'fc':
				layer = FCLayer(input_size, layer_def['n'], layer_def['act_type'])
				input_size = (layer.n_neurons,)
			# Maxpool layer
			elif layer_def['layer'] == 'maxpool':
				layer = MaxPoolLayer(input_size, (layer_def['kern_size'], layer_def['kern_size']), layer_def['stride'])
				input_size = layer.output_shape
			# Input size definition
			elif layer_def['layer'] == 'input':
				input_size = (layer_def['c'], layer_def['h'], layer_def['w'])
				self.input_size = input_size
				print "* Input -> Size: ", input_size
				continue
			# Error layer
			elif layer_def['layer'] == 'error':
				layer = ErrorLayer(layer_def['type'])
			else:
				raise ValueError("*** UNKOWN LAYER TYPE ("+layer_def['layer']+")")
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
		
		weights = empty([0])
		for layer in self.layers:
			if layer.has_weights:
				weights = np.append(weights, layer.weights.ravel())
				weights = np.append(weights, layer.bias.ravel())

		# Saving the weights
		with open(file_path, 'w') as f:
		    #pickle.dump(weights, f)
		    np.save(f, weights, allow_pickle=False)

		#weights.tofile(file_path)

		print "* NET: Saved my weights! (to: "+file_path+", " + "%.3f mb)"%(len(weights)*8.0/1000000.0)

	def load_weights(self, file_path):
		"""
		Loads the weights from a pickle file
		(make sure thats the same net, i dont really check that)
		"""
		print "* NET: Loading my weights! (from: "+file_path+")"
		# Getting back the weights
		with open(file_path) as f:
		    #weights = pickle.load(f)
		    weights = np.load(f)

		#weights = np.fromfile(file_path, dtype=np.float64)

		p = 0
		for layer in self.layers:
			if layer.has_weights:
				n = np.prod(layer.weights.shape)
				l_weights = weights[p:p+n]
				layer.weights = l_weights.reshape(layer.weights.shape)
				p += n
				nb = np.prod(layer.bias.shape)
				l_bias = weights[p:p+nb]
				layer.bias = l_bias.reshape(layer.bias.shape)
				p += nb


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

		# Visualize your weights with openCV
		# if False:
		# 	weights = net.check_weights(0).transpose(1,2,0)
		# 	weights = cv2.resize(weights,  (0,0), fx=20, fy=20, interpolation=cv2.INTER_AREA)

		# 	cv2.imshow('weights',weights)
		# 	cv2.imwrite('weights_0.png',weights)
		# 	cv2.waitKey(0)



		return weights

	#def __del__(self):