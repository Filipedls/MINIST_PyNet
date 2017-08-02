import numpy as np
from get_cifar import *

from context import pynet

# A quick 6 epochs run on cifar-10 with about 80% accuracy

#import mkl
#mkl.set_num_threads(2)

cifar_ds = get_cifar_10('train')

cifar_ds_test = get_cifar_10('test')

weights_f_name = 'weights_back_cifar.weights'
load_weights_from_file = True

config = {
	'train_set' : cifar_ds,
	'test_set' : cifar_ds_test,
	'ds_mean_std' : [[ 125.30691, 122.95039, 113.865383], [ 51.56153, 50.82543, 51.22018]],
	'print_every_itr': 100,
	'type' : "momentum",
	'params' : {
		'lr' : [0.0003, [100, 10.0], [1000,0.5], [4000, 0.5], [10000, 0.5], [19000,0.1]], # starting_value,...,[iter,multiplier],...
		'batch': [16],
		'w_decay' : [0.000001],
		'momentum' : [0.9],
		'max_iter' : 20000
	},

	'save_every_itr': 2000,
	'save_file_name' : weights_f_name
}

net_def =  [
			{'layer': 'input', 'c': 3, 'h': 32, 'w': 32},
			{'layer': 'conv', 'kern_size': 3, 'n': 32, 'pad': 0, 'stride': 1},
			{'layer': 'conv', 'kern_size': 3, 'n': 32, 'pad': 0, 'stride': 1},
			{'layer': 'maxpool', 'kern_size': 2, 'stride': 2},
			{'layer': 'conv', 'kern_size': 3, 'n': 32, 'pad': 0, 'stride': 1},
			{'layer': 'conv', 'kern_size': 3, 'n': 64, 'pad': 0, 'stride': 1},
			{'layer': 'maxpool', 'kern_size': 2, 'stride': 2},
			{'layer': 'fc', 'n': 64, 'act_type': 'lerelu'},
			{'layer': 'fc', 'n': 10, 'act_type': 'softmax'},
			{'layer': 'error', 'type': 'l1'}
		   ]


net = pynet.Net(net_def)

if load_weights_from_file:
	net.load_weights(weights_f_name)

trainer = pynet.Trainer(net, config)

if load_weights_from_file:
	trainer.test()
elif trainer.train():
	trainer.net.save_weights(weights_f_name)
	trainer.test()
else:
	print "Train was not sucessful... :("

