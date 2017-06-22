from net import *
from helpers import *
import random
import cv2
from timeit import default_timer as timer

class Trainer:

	def __init__( self, net, config):
		print "* TRAINER CONFIG: ", config

		self.net = net

		self.training_set = self.get_set_from_txt(config['train_set'])
		self.training_n = len(self.training_set)
		self.training_dir = config['train_dir']
		print "* TRAINER: ", len(self.training_set), " images found for train."

		self.test_set = self.get_set_from_txt(config['test_set'])
		self.test_n = len(self.test_set)
		self.test_dir = config['test_dir']
		print "* TRAINER: ", len(self.test_set), " images found for test."

		if config.has_key('ds_mean_std'):
			self.mean = config['ds_mean_std'][0][0]
			self.std = config['ds_mean_std'][1][0]
			print "* Dataset mean found: ", self.mean, "; std: ", self.std
		else:
			self.mean, self.std = self.get_dataset_mean_std()
			print "* Dataset mean checked: ", self.mean, "; std: ", self.std

		self.print_every_itr = config['print_every_itr']

	def train(self, max_iter, learning_rate, batch_size):

		print "Training ", self.training_n, " images; batch: ", batch_size

		train_samples_idx = range(0,self.training_n)
		random.shuffle(train_samples_idx)

		error = 0.0
		epoch = 0
		n_samples = 0
		time = 0.0
		time_f = 0.0
		for iter in range(1, max_iter+1):

			for i_batch in range(0,batch_size):
				i = (iter * batch_size + i_batch) % self.training_n
				# Input
				train_sample = self.training_set[train_samples_idx[i]]
				path = self.training_dir + train_sample['path']
				input = self.preprocess_img(path, self.mean, self.std)

				label = zeros(self.net.n_classes)
				label[train_sample['class']] = 1

				start = timer()
				error += self.net.forward(input, label)
				time_f += timer() - start
				self.net.backward()
				time += timer() - start

				n_samples += 1
				if i == self.training_n-1:
					epoch += 1
					random.shuffle(train_samples_idx)

			# Update the weights at the end of every batch
			scale = self.net.net_checks(learning_rate/batch_size)
			self.net.update_weights(learning_rate/batch_size)

			if iter % self.print_every_itr == 0:
				print_iter_n = (batch_size*self.print_every_itr)
				print iter,"\tE: %.3f"% (error/print_iter_n), "lr:", learning_rate,"\tN:",n_samples,"\tEp:",epoch, "\tT: %.1f %.1f ms" % (time_f*1000/print_iter_n, time*1000/print_iter_n)," (",scale,")"
				error = 0.0
				scale = 0
				time = 0.0
				time_f = 0.0


			if iter == 10:
				learning_rate = 10*learning_rate

			if iter > 5000:
				learning_rate = 0.9992*learning_rate


	def test(self):

		print "Testing ", self.test_n, " images"

		error = 0.0
		n_samples = 0
		right = 0
		time = 0.0
		for i in range(0,self.test_n):

			# Input
			test_sample = self.test_set[i]
			path = self.test_dir + test_sample['path']
			input = self.preprocess_img(path, self.mean, self.std)

			label = zeros(self.n_classes)
			label[test_sample['class']] = 1

			start = timer()
			error += self.net.forward(input, label)
			time += timer() - start

			pred = self.layers[-1].input

			if test_sample['class'] == np.argmax(pred):
				right += 1

			n_samples += 1


		print "* TEST" ,"\tA: %.3f"%(right/float(n_samples)),"\tE: %.3f"% (error/self.test_n), "\tN:",n_samples, "\tT: %.1F ms" % (time*1000/self.test_n)


	def get_dataset_mean_std(self):

		input = cv2.imread(self.training_dir + self.training_set[0]['path'], 4)#, cv2.IMREAD_GRAYSCALE)
		if len(input.shape) == 3:
			mean = np.zeros(3)
			std = np.zeros(3)
		else:
			mean = 0.0
			std = 0.0

		for imgPath in self.training_set:
			# Input
			path = self.training_dir + imgPath['path']
			input = cv2.imread(path, 4)#, cv2.IMREAD_GRAYSCALE)

			mean += np.mean(input, axis = (0,1))
			std += np.std(input, axis = (0,1))

		mean = mean / self.training_n
		std = std / self.training_n

		print "* Dataset mean: ", mean, "; std: ", std
		return mean, std

	def preprocess_img(self, path, mean, std):

		img = cv2.imread(path, 4)
		input = cv2.resize(img, tuple(self.net.input_size[1:3]), interpolation=cv2.INTER_AREA)
		if self.net.input_size[0] == 1:#len(input.shape) == 2:
			input = np.expand_dims(input, axis=0)
		else:
			input = np.transpose(input, (2,0,1))

		input = (input - mean)/std

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