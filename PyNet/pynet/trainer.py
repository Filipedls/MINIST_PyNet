from net import *
from updater import *
from helpers import *
import random
import cv2 # TODO: do we really need this here?!
from timeit import default_timer as timer

class Trainer:

	def __init__( self, net, config):
		"""
		Initializes a Trainer, to train a net
		"""

		self.net = net

		# SETS THE TRAINING SET
		if config.has_key('train_set'):
			if isinstance(config['train_set'], str):
				self.training_set = self.get_set_from_txt(config['train_set'])
				self.training_dir = config['train_dir']
				self.input_from_dir = True
			else:
				self.training_set = config['train_set']
				self.input_from_dir = False

			self.training_n = len(self.training_set)
			print "* TRAINER: ", len(self.training_set), " images found for train, dir:", self.input_from_dir

		else:
			self.training_n = 0
			print "* TRAINER: NO TRAINIG DATASET FOUND!!!"

		# SETS THE TEST SET
		if config.has_key('test_set'):
			if self.input_from_dir:
				self.test_set = self.get_set_from_txt(config['test_set'])
				self.test_dir = config['test_dir']
			else:
				self.test_set = config['test_set']

			self.test_n = len(self.test_set)

			print "* TRAINER: ", len(self.test_set), " images found for test."

		# Dataset mean and std
		if config.has_key('ds_mean_std'):
			self.mean = config['ds_mean_std'][0]
			self.std = config['ds_mean_std'][1]
			print "* Dataset mean found: ", self.mean, "; std: ", self.std
		else:
			self.mean = 0.0
			self.std = 1.0
			self.mean, self.std = self.get_dataset_mean_std()
			print "* Dataset mean checked: ", self.mean, "; std: ", self.std

		self.print_every_itr = config['print_every_itr']

		# The algorithm used to optimize
		if config.has_key('type'):
			self.type = config['type']
		else:
			self.type = 'sgd'

		self.updater = Updater(net, self.type)

		self.params = config['params']

		if config.has_key('save_every_itr'):
			self.save_iter = config['save_every_itr']
			self.back_file_name = config['save_file_name']
		else:
			self.save_iter = 0


	def train(self):
		"""
		Trains a net with the parameters initialized
		"""

		if self.training_n == 0:
			raise ValueError("* NO TRAINING IMAGES!")

		print "Training", self.training_n, "images; batch:", self.params['batch'][0], "; lr:", self.params['lr'][0], "; w_decay:", self.params['w_decay'][0], "; momentum:",self.params['momentum'][0]
		# First shufle of the training data
		train_samples_idx = range(0,self.training_n)
		random.shuffle(train_samples_idx)

		error = 0.0
		btch_error = 0.0
		epoch = 0
		n_samples = 0
		time_b = 0.0
		time_f = 0.0
		scale = 0.0
		for iter in range(1, self.params['max_iter']+1):
			inputs = empty((self.params['batch'][0],)+self.net.input_size)
			labels = zeros((self.params['batch'][0],self.net.n_classes))

			for i_batch in range(0,self.params['batch'][0]):
				i = ((iter-1) * self.params['batch'][0] + i_batch) % self.training_n
				# Input
				input, input_class = self.get_input(train_samples_idx[i], 'train')
				
				labels[i_batch,input_class] = 1
				inputs[i_batch,:,:,:] = input

				n_samples += 1
				if i == self.training_n-1:
					epoch += 1
					print epoch, ' Epochs','; E:', (btch_error / self.training_n)
					btch_error = 0.0
					random.shuffle(train_samples_idx)

			start_f = timer()
			iter_error = self.net.forward(inputs, labels)
			error += iter_error
			btch_error += iter_error
			time_f += timer() - start_f
			start_b = timer()
			self.net.backward()
			time_b += timer() - start_b

			# Update the weights at the end of every batch
			self.updater.update_weights(self.params)
			scale += self.updater.net_checks()
			# Printing training stuff
			if iter % self.print_every_itr == 0:
				print_iter_n = self.params['batch'][0]*self.print_every_itr
				print iter,"\tE: %.2f"% (error/print_iter_n), "lr:", self.params['lr'][0],"\tN:",n_samples, "\tF/B %.1f/%.1f (ms)" % (time_f*1000/self.print_every_itr, time_b*1000/self.print_every_itr),"(%.2f"%(scale*1000/self.print_every_itr)+")"
				error = 0.0
				time_b = 0.0
				time_f = 0.0
				scale = 0.0
			# saving the weights
			if self.save_iter and iter % self.save_iter == 0:
				self.net.save_weights(self.back_file_name)
				print "Weights saved at "+self.back_file_name

			self.check_params(iter)

		return True

	def test(self):
		"""
		Tests a net...
		"""

		print "Testing ", self.test_n, " images"

		error = 0.0
		n_samples = 0
		right = 0
		time = 0.0
		for i in range(self.test_n):

			# Input
			#input, input_class = self.get_input(train_samples_idx[i])

			#test_sample = self.test_set[i]
			#path = self.test_dir + test_sample['path']
			input, class_n = self.get_input(i, 'test')
			input = np.expand_dims(input, axis=0) # batch like format
			label = zeros((1,self.net.n_classes))
			label[0,class_n] = 1

			start = timer()
			error += self.net.forward(input, label)
			time += timer() - start

			pred = self.net.layers[-1].input

			if class_n == np.argmax(pred):
				right += 1

			n_samples += 1


		print "* TEST" ,"\tA: %.3f"%(right/float(n_samples)),"\tE: %.3f"% (error/self.test_n), "\tN:",n_samples, "\tT: %.1F ms" % (time*1000/self.test_n)

	def check_params(self, train_iter):
		"""
		Checks and, if needed, updates the training parameters
		"""
		for name, value in self.params.iteritems(): # for all the training parameters
			if isinstance(value, list) and len(value) > 1: # if its not a constant parameter
				if isinstance(value[1][0], int): # step parameter
					for iter, parm_value in value[1:]: # checks all steps
						if iter == train_iter: # and if one of the iter matches
							value[0] = value[0]*parm_value # multiplies the current value with it
				else:
					print "* check_params: unknown parameter!"


	def get_dataset_mean_std(self):
		"""
		Checks and, if needed, updates the training parameters
		"""

		input, cls = self.get_input(0)#, cv2.IMREAD_GRAYSCALE)
		print input.shape
		if len(input.shape) == 3:
			mean = np.zeros(3)
			std = np.zeros(3)
		else:
			mean = 0.0
			std = 0.0

		for idx in range(self.training_n):
			# Input
			input, cls = self.get_input(idx)

			mean += np.mean(input, axis = (1,2))
			std += np.std(input, axis = (1,2))

		mean = mean / self.training_n
		std = std / self.training_n

		print "* Dataset mean: ", mean, "; std: ", std
		return mean, std

	def get_input(self, sample_idx, from_set):

		if from_set == 'train':
			input_set = self.training_set
			if self.input_from_dir:
				set_dir = self.training_dir
		elif from_set == 'test':
			input_set = self.test_set
			if self.input_from_dir:
				set_dir = self.test_dir
		else:
			print '* get_input(): DONT KNOW THAT SET!'

		if self.input_from_dir:
			input_sample = input_set[sample_idx]
			path = set_dir + input_sample['path']
			input_raw = self.get_img_from_dir(path)
			input_class = input_sample['class']
		else:
			input_raw = self.training_set[sample_idx][0]
			input_class =  self.training_set[sample_idx][1]

		input_raw = astype(input_raw)
		input = self.preprocess_img(input_raw, self.mean, self.std)
		return input, input_class

	def get_img_from_dir(self, path):
		input = cv2.imread(path, 4)

		if len(input.shape) == 2:
			input = np.expand_dims(input, axis=0)
		else:
			input = input.transpose((2,0,1)) # numpy style, chanels as the 0th axis
		return input

	def preprocess_img(self, input, mean, std):

		if input.shape[1:3] != self.net.input_size[1:3]:
			input = cv2.resize(input, tuple(self.net.input_size[1:3]), interpolation=cv2.INTER_AREA)

		for i in range(self.net.input_size[0]):
			input[i,:,:] = (input[i,:,:] - mean[i])/std[i]

		#input = input/(255.0/2.0) - 1.0

		return input


	def get_set_from_txt(self, path):
		set_l = []
		with open(path,"r") as f:
			lines = f.readlines()

		class_lim = 50000
		class_cnt = {}
		random.shuffle(lines)
		for line in lines:
			path, class_n = line.split(" ")
			class_n = int(class_n)

			if class_cnt.has_key(class_n):
				class_cnt[class_n] += 1
			else:
				class_cnt[class_n] = 1

			if class_cnt[class_n] <= class_lim:
				set_l.append({'path' : path, 'class' : class_n}) 

		return set_l

	#def __del__(self):