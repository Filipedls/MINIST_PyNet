from net import *
from trainer import *
from helpers import *
import numpy as np
import cv2
import math
import random


config = {
	'train_set' : "../../mnist_png/train.txt",
	'train_dir' : "../../mnist_png/training/",
	'test_set' : "../../mnist_png/test.txt",
	'test_dir' : "../../mnist_png/testing/",
	'ds_mean_std' : [[33.32],[76.83]],
	'print_every_itr': 1

}

net_def_basic =  [['input', 1, 20, 20],
			['fc', 400, 'relu'],
			['fc', 27, 'relu'],
			['fc', 10, 'softmax'],
			['error','l1']
			]


net_def =  [['input', 1, 28, 28],
			['conv', 3, 32, 0, 1],
			['conv', 3, 64, 0, 1],
			['maxpool', 2, 2],
			['conv', 3, 128, 0, 1],
			['conv', 3, 256, 0, 1],
			['conv', 3, 512, 0, 1],
			['maxpool', 2, 2],
			['fc', 2048, 'lerelu'],
			['fc', 1024, 'lerelu'],
			['fc', 256, 'lerelu'],
			['fc', 10, 'softmax'],
			['error','l1']
		   ]


net = Net(net_def)

trainer = Trainer(net, config)

num_iter = 1000
trainer.train( num_iter, 0.0005, 16)

trainer.test()


#print "CHECKS 1: input: \n", net.layers[9].input, "; d_input: \n", net.layers[9].output_error
#print "CHECKS 2: input: ", net.layers[8].weights[0,:,:,:], "; d_input: ", net.layers[8].d_weights[0,:,:,:] 