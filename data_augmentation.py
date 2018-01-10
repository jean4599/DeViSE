import get_data
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

def generate():

	# load img
	(train_data, train_coarse_labels), (eval_data, eval_coarse_labels) = get_data.load_data(label_mode='coarse', path='./Data/cifar-100/')
	(train_data, train_fine_labels), (eval_data, eval_fine_labels) = get_data.load_data(label_mode='fine', path='./Data/cifar-100/')

	classes = get_data.unpickle('./Data/cifar-100/meta')
	fine_class = np.asarray(classes[b'fine_label_names'], dtype=np.dtype(np.str))
	course_class = np.asarray(classes[b'coarse_label_names'], dtype=np.dtype(np.str))

	datagen = ImageDataGenerator(rotation_range=0.2, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

	x_train = train_data
	y_train = train_fine_labels


	for x_batch, y_batch in datagen.flow(train_data, train_fine_labels, batch_size=200):
		x_train = np.concatenate((x_train, x_batch), axis=0)
		y_train = np.concatenate((y_train, y_batch), axis=0)
		if x_train.shape[0] == 50000*2:
			break

	return (x_train, y_train), (eval_data, eval_fine_labels)

