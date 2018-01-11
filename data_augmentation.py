import get_data
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

def generate(label_mode='fine'):

	# load img
	(x_train, y_train_fine_label, y_train_coarse_label), (x_test, y_test_fine_label, y_test_coarse_label) = get_data.load_data(label_mode='both', path='./Data/cifar-100/')

	datagen = ImageDataGenerator(rotation_range=0.2, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

	y_train = []
	y_train_fine_label = np.array([y_train_fine_label])
	y_train_fine_label = y_train_fine_label.reshape(y_train_fine_label.shape[1], y_train_fine_label.shape[0])
	y_train_coarse_label = np.array([y_train_coarse_label])
	y_train_coarse_label = y_train_coarse_label.reshape(y_train_coarse_label.shape[1], y_train_coarse_label.shape[0])
	y_train = np.concatenate((y_train_fine_label, y_train_coarse_label), axis=1)


	for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=200):
		x_train = np.concatenate((x_train, x_batch), axis=0)
		y_train = np.concatenate((y_train, y_batch), axis=0)
		print('x_train shape', x_train.shape)
		if x_train.shape[0] >= 50000*2:
			break

	(y_train_fine, y_train_coarse) = np.split(y_train, 2, axis=1)
	y_train_fine = y_train_fine.reshape(y_train_fine.shape[0])
	y_train_coarse = y_train_coarse.reshape(y_train_coarse.shape[0])


	if label_mode=='fine':
		return (x_train, y_train_fine), (x_test, y_test_fine_label)
	elif label_mode=='both':
		return (x_train, y_train_fine, y_train_coarse), (x_test, y_test_fine_label, y_test_coarse_label)

