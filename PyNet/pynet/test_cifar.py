import numpy as np
from get_cifar import *

#np.seterr(all='print')

#import mkl
#mkl.set_num_threads(2)

cifar_ds = get_cifar_10('train')

cifar_ds_test = get_cifar_10('test')

config = {
	'train_set' : cifar_ds,
	'test_set' : cifar_ds_test,
	'ds_mean_std' : [[ 125.30691805, 122.95039414, 113.86538318], [ 51.56153984, 50.82543151, 51.22018275]],
	'print_every_itr': 100,
	'type' : "momentum"

}

net_def =  [['input', 3, 32, 32],
			['conv', 3, 16, 0, 1],
			['conv', 3, 32, 0, 1],
			['maxpool', 2, 2],
			['conv', 3, 64, 0, 1],
			['maxpool', 2, 2],
			#['conv', 3, 64, 0, 1],
			#['maxpool', 2, 2],
			['fc', 200, 'lerelu'],
			['fc', 10, 'softmax'],
			['error','l1']
		   ]


net = Net(net_def)

trainer = Trainer(net, config)

num_iter = 40000
if trainer.train( num_iter, 0.0005, 32, 0.0000):

	trainer.test()
