
from context import pynet


config = {
	'train_set' : "../../mnist_png/train.txt",
	'train_dir' : "../../mnist_png/training/",
	'test_set' : "../../mnist_png/test.txt",
	'test_dir' : "../../mnist_png/testing/",
	'ds_mean_std' : [[33.32],[76.83]],
	'print_every_itr': 100

}

net_def_basic =  [['input', 1, 20, 20],
			['fc', 400, 'relu'],
			['fc', 27, 'relu'],
			['fc', 10, 'softmax'],
			['error','l1']
			]


net_def =  [['input', 1, 28, 28],
			['conv', 3, 8, 0, 1],
			['conv', 3, 16, 0, 1],
			['maxpool', 2, 2],
			['conv', 3, 24, 0, 1],
			['conv', 3, 32, 0, 1],
			['maxpool', 2, 2],
			['fc', 512, 'relu'],
			['fc', 64, 'relu'],
			['fc', 10, 'softmax'],
			['error','l1']
			]


net = pynet.Net(net_def, config)


num_iter = 10000
net.train( num_iter, 0.0005, 16)

net.test()