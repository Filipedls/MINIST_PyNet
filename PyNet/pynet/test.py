from net import *
from trainer import *

# A one epoch run on MINST with about 98% accuracy

weights_f_name = 'weights_back_MNIST.pickle'

load_weights_from_file = True

config = {
	'train_set' : "../../mnist_png/train.txt",
	'train_dir' : "../../mnist_png/training/",
	'test_set' : "../../mnist_png/test.txt",
	'test_dir' : "../../mnist_png/testing/",
	'ds_mean_std' : [[33.32],[76.83]],
	'print_every_itr': 100,
	'type' : "momentum",
	'params' : {
		'lr' : [0.0003, [100, 10.0], [1000, 0.5], [2000, 0.5], [3250,0.1]], # starting_value,...,[iter,multiplier],...
		'batch': [16],
		'w_decay' : [0.000001],
		'momentum' : [0.9],
		'max_iter' : 3750
	},

}

net_def_basic =  [['input', 1, 20, 20],
			['fc', 400, 'relu'],
			['fc', 27, 'relu'],
			['fc', 10, 'softmax'],
			['error','l1']
			]


net_def =  [['input', 1, 28, 28],
			['conv', 3, 8, 0, 1],
			['conv', 3, 8, 0, 1],
			['maxpool', 2, 2],
			['conv', 3, 16, 0, 1],
			['conv', 3, 16, 0, 1],
			['maxpool', 2, 2],
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