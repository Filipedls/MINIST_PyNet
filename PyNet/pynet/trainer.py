from net import *
from updater import *
from helpers import *
import random
import cv2
from timeit import default_timer as timer

class Trainer:

	def __init__( self, net, config):
		#print "* TRAINER CONFIG: ", config

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


	def train(self, max_iter, learning_rate, batch_size, w_decay):

		if self.training_n == 0:
			raise ValueError("* NO TRAINING IMAGES!")

		print "Training", self.training_n, "images; batch:", batch_size, "; lr:", learning_rate, "; w_decay:", w_decay

		train_samples_idx = range(0,self.training_n)
		random.shuffle(train_samples_idx)

		mom = 0.9

		error = 0.0
		epoch = 0
		n_samples = 0
		time_b = 0.0
		time_f = 0.0
		time_btch = 0.0
		scale = 0.0
		for iter in range(1, max_iter+1):

			start_btch = timer()
			for i_batch in range(0,batch_size):
				i = (iter * batch_size + i_batch) % self.training_n
				# Input
				input, input_class = self.get_input(train_samples_idx[i])
				#print "INPUT:\n",np.max(input),np.min(input)
				label = zeros(self.net.n_classes)
				label[input_class] = 1

				start_f = timer()
				error += self.net.forward(input, label)
				time_f += timer() - start_f
				start_b = timer()
				self.net.backward()
				time_b += timer() - start_b

				n_samples += 1
				if i == self.training_n-1:
					epoch += 1
					random.shuffle(train_samples_idx)

			time_btch += timer() - start_btch

			# Update the weights at the end of every batch
			self.updater.update_weights(learning_rate/batch_size, mom, w_decay*batch_size)
			iter_scale = self.updater.net_checks()
			scale += iter_scale
			
			if iter % self.print_every_itr == 0:
				
				print_iter_n = (batch_size*self.print_every_itr)
				print iter,"\tE: %.2f"% (error/print_iter_n), "lr:", learning_rate,"\tN:",n_samples,"\tEp:",epoch, "\tF/B/I %.1f/%.1f/%.1f (ms)" % (time_f*1000/print_iter_n, time_b*1000/print_iter_n, time_btch*1000/self.print_every_itr)," (%.2f"%(scale*1000/self.print_every_itr),")"
				error = 0.0
				scale = 0
				time_b = 0.0
				time_f = 0.0
				time_btch = 0.0
				scale = 0.0

			if iter == 100:
				learning_rate = 10*learning_rate
				#batch_size = int(batch_size*0.5)
				#mom = 0.75

			if iter == 5000:
				learning_rate = 0.5*learning_rate
				batch_size = 75

			if iter == 25000 or iter  == 29000:
				learning_rate = 0.1*learning_rate
				#batch_size = int(batch_size*2)
				#mom = mom+0.4

			if np.isnan(error):
				print "ABORTED : ERROR TO BIG!"
				return False
			#if iter_scale*1000 > 5:
			#	learning_rate = 0.5*learning_rate

		return True

	def test(self):

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
			input, class_n = self.test_set[i]#self.preprocess_img(self.get_img_from_dir(path), self.mean, self.std)
			input = astype(input)
			input = self.preprocess_img(input, self.mean, self.std)
			label = zeros(self.net.n_classes)
			label[class_n] = 1

			start = timer()
			error += self.net.forward(input, label)
			time += timer() - start

			pred = self.net.layers[-1].input

			if class_n == np.argmax(pred):
				right += 1

			n_samples += 1


		print "* TEST" ,"\tA: %.3f"%(right/float(n_samples)),"\tE: %.3f"% (error/self.test_n), "\tN:",n_samples, "\tT: %.1F ms" % (time*1000/self.test_n)


	def get_dataset_mean_std(self):

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

	def get_input(self, train_sample_idx):
		if self.input_from_dir:
			train_sample = self.training_set[train_sample_idx]
			path = self.training_dir + train_sample['path']
			input = self.preprocess_img(self.get_img_from_dir(path), self.mean, self.std)
			input_class = train_sample['class']
		else:
			input_raw = astype(self.training_set[train_sample_idx][0])
			input = self.preprocess_img(input_raw, self.mean, self.std)
			input_class =  self.training_set[train_sample_idx][1]

		input = astype(input)
		return input, input_class

	def get_img_from_dir(self, path):
		return cv2.imread(path, 4)

	def preprocess_img(self, input, mean, std):

		if input.shape[1:3] != self.net.input_size[1:3]:
			input = cv2.resize(input, tuple(self.net.input_size[1:3]), interpolation=cv2.INTER_AREA)
		
		if self.net.input_size[0] == 1:#len(input.shape) == 2:
			input = np.expand_dims(input, axis=0)
		elif self.input_from_dir:
			input = np.transpose(input, (2,0,1))

		#
		input[0,:,:] = (input[0,:,:] - mean[0])/std[0]
		input[1,:,:] = (input[1,:,:] - mean[1])/std[1]
		input[2,:,:] = (input[2,:,:] - mean[2])/std[2]

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