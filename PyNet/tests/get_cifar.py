import numpy as np

batches_dir = '../../cifar-10-batches-py/'

test_batch = ('test_batch',)
train_batches = ('data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5')

batch_size = 10000
img_shape = (3,32,32)

def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

cifar_meta = unpickle(batches_dir+'batches.meta')

labels_dic = cifar_meta['label_names']

batch_size = cifar_meta['num_cases_per_batch']
img_vec_len = cifar_meta['num_vis']

def get_cifar_10(ds_type):
	if ds_type == 'train':
		data_batches = train_batches
	else:
		data_batches = test_batch

	#images = np.empty((len(data_batches)*batch_size,img_vec_len), dtype=np.uint8)
	#labels = np.empty((len(data_batches)*batch_size))

	dataset = []

	class_lim = 10000
	class_count = np.zeros(10)

	for i in range(len(data_batches)):
		dict = unpickle('../../cifar-10-batches-py/'+data_batches[i])

		#images[i*batch_size:(i+1)*batch_size,:] = dict['data']
		#labels[i*batch_size:(i+1)*batch_size] = dict['labels']
		for img_n in range(len(dict['labels'])):

			img = dict['data'][img_n,:].reshape(3,32,32)

			label_n = dict['labels'][img_n]

			# TO bgr
			#img = img[[2,1,0],:,:]
			# switch dims for cv2
			#img = img.transpose((1,2,0))

			class_count[label_n] += 1
			if class_count[label_n] <= class_lim:
				dataset.append((img,label_n))
			
	print len(dataset)," cifar images found"

	return dataset
