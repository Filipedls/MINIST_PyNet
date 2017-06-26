import numpy as np
from get_cifar import *

#np.seterr(all='print')

cifar_ds = get_cifar_10('train')

cifar_ds_test = get_cifar_10('test')

config = {
	'train_set' : cifar_ds,
	'test_set' : cifar_ds_test,
	'ds_mean_std' : [[ 125.30691805, 122.95039414, 113.86538318], [ 51.56153984, 50.82543151, 51.22018275]],
	'print_every_itr': 10

}

net_def =  [['input', 3, 32, 32],
			['conv', 5, 18, 0, 1],
			['maxpool', 2, 2],
			['conv', 5, 38, 0, 1],
			['maxpool', 2, 2],
			['fc', 768, 'lerelu'],
			['fc', 256, 'lerelu'],
			['fc', 10, 'softmax'],
			['error','l1']
		   ]


net = Net(net_def)

trainer = Trainer(net, config)

num_iter = 10
trainer.train( num_iter, 0.001, 4)

trainer.test()
