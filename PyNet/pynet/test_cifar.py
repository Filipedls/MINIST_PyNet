import numpy as np
import cv2

data_batches = ('data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5')
test_batch = 'test_batch'

batch_size = 10000


def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict


labels_dic = unpickle('../../cifar-10-batches-py/batches.meta')

images = np.empty((len(data_batches)*batch_size,3072), dtype=np.uint8)
labels = np.empty((len(data_batches)*batch_size))

dataset = []

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

		dataset.append((img,label_n))


print len(dataset)," images found"

print dataset[0]


#cv2.imwrite(str(img_n)+'.png',img)
#cv2.imshow('', img)
#cv2.waitKey()
#cv2.destroyAllWindows()