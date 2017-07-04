import numpy as np
from get_cifar import *

#np.seterr(all='print')

#import mkl
#mkl.set_num_threads(2)

cifar_ds = get_cifar_10('train')

cifar_ds_test = get_cifar_10('test')

weights_f_name = 'weights_back.pickle'

config = {
	'train_set' : cifar_ds,
	'test_set' : cifar_ds_test,
	'ds_mean_std' : [[ 125.30691805, 122.95039414, 113.86538318], [ 51.56153984, 50.82543151, 51.22018275]],
	'print_every_itr': 100,
	'type' : "momentum",
	'params' : {
		'lr' : [0.0003, [100,10.0], [3000,0.5], [7000, 0.5], [11000, 0.5], [15000, 0.5], [19000,0.1]], # starting_value,...,[iter,multiplier],...
		'batch': [100],
		'w_decay' : [0.000001],
		'momentum' : [0.9],
		'max_iter' : 20000
	},

	'save_every_itr': 1000,
	'save_file_name' : weights_f_name
}

net_def =  [['input', 3, 32, 32],
			['conv', 3, 24, 0, 1],
			['maxpool', 3, 3],
			['conv', 3, 48, 0, 1],
			['maxpool', 2, 2],
			#['conv', 5, 64, 2, 1],
			#['maxpool', 2, 2],
			#['conv', 3, 64, 0, 1],
			#['maxpool', 2, 2],
			#['fc', 512, 'lerelu'],
			['fc', 76, 'lerelu'],
			['fc', 10, 'softmax'],
			['error','l1']
		   ]


net = Net(net_def)
net.load_weights(weights_f_name)

trainer = Trainer(net, config)
trainer.test()

#if trainer.train( ):
#	net.save_weights(weights_f_name)
#	trainer.test()
