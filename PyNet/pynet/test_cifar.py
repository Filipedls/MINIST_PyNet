import numpy as np
from get_cifar import *
import cv2

# A quick 3 epochs run on cifar-10 with about 70% accuracy
# If you load the weights you only get 50%... probably some lost of precision while saving them

#import mkl
#mkl.set_num_threads(2)

cifar_ds = get_cifar_10('train')

cifar_ds_test = get_cifar_10('test')

weights_f_name = 'weights_back_cifar.pickle'
load_weights_from_file = True

config = {
	'train_set' : cifar_ds,
	'test_set' : cifar_ds_test,
	'ds_mean_std' : [[ 125.30691, 122.95039, 113.865383, [ 51.56153, 50.82543, 51.22018]],
	'print_every_itr': 100,
	'type' : "momentum",
	'params' : {
		'lr' : [0.0003, [100, 10.0], [1000,0.5], [4000, 0.5], [20000, 0.5], [29000,0.1]], # starting_value,...,[iter,multiplier],...
		'batch': [32],
		'w_decay' : [0.000001],
		'momentum' : [0.9],
		'max_iter' : 5000
	},

	'save_every_itr': 2000,
	'save_file_name' : weights_f_name
}

net_def =  [['input', 3, 32, 32],
			['conv', 3, 32, 0, 1],
			['conv', 3, 32, 0, 1],
			['maxpool', 2, 2],
			['conv', 3, 32, 0, 1],
			['conv', 3, 64, 0, 1],
			['maxpool', 2, 2],
			#['fc', 512, 'lerelu'],
			['fc', 64, 'lerelu'],
			['fc', 10, 'softmax'],
			['error','l1']
		   ]


net = Net(net_def)

if load_weights_from_file:
	net.load_weights(weights_f_name)

trainer = Trainer(net, config)

if load_weights_from_file:
	trainer.test()
elif trainer.train():
	net.save_weights(weights_f_name)
	trainer.test()
else:
	print "Train was not sucessful... :("

# Visualize your weights with openCV
if False:
	weights = net.check_weights(0).transpose(1,2,0)
	weights = cv2.resize(weights,  (0,0), fx=20, fy=20, interpolation=cv2.INTER_AREA)

	cv2.imshow('weights',weights)
	cv2.imwrite('weights_0.png',weights)
	cv2.waitKey(0)



