from net import *
from helpers import *
import numpy as np

class Updater:

	def __init__( self, net, type):
		if type == 'sgd':
			self.update_weights = self.udpate_sgd
		elif type == 'momentum':
			self.w_velocities = []
			self.b_velocities = []
			for layer in net.layers:
				if layer.has_weights:
					self.w_velocities.append(zeros(layer.weights.shape))
					self.b_velocities.append(zeros(layer.bias.shape))

			self.update_weights = self.udpate_mom

		self.net = net


	def udpate_mom(self, lr, mu):
		i = 0
		for layer in self.net.layers:
			if layer.has_weights:
				# Velocities
				self.w_velocities[i] = mu * self.w_velocities[i] - lr * layer.d_weights
				self.b_velocities[i] = mu * self.b_velocities[i] - lr * layer.d_bias
				# Update weights
				layer.weights += self.w_velocities[i]
				layer.bias += self.b_velocities[i]
				# Reset the derivatives
				layer.d_weights = zeros(layer.weights.shape)
				layer.d_bias = zeros(layer.bias.shape)
				i += 1



	def udpate_sgd(self, lr, mu):
		for layer in self.net.layers:
			if layer.has_weights:
				# Update weights
				layer.weights -= layer.d_weights * lr
				layer.bias -= layer.d_bias * lr
				# Reset the derivatives
				layer.d_weights = zeros(layer.weights.shape)
				layer.d_bias = zeros(layer.bias.shape)

	def net_checks(self):

		scale = 0
		n = 0
		for layer in self.net.layers:
			if layer.has_weights:
				weights = np.concatenate([layer.weights.ravel(), layer.bias.ravel()])
				param_scale = np.linalg.norm(weights)
				update = np.concatenate([self.w_velocities[n].ravel(), self.b_velocities[n].ravel()])
				#update = -learning_rate * layer.d_weights
				update_scale = np.linalg.norm(update.ravel())
				n += 1
				scale += update_scale / param_scale

		return scale/n
	#def __del__(self):