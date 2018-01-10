import get_data
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(rotation_range=0.2, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

# load img
(train_data, train_coarse_labels), (eval_data, eval_coarse_labels) = get_data.load_data(label_mode='coarse', path='./Data/cifar-100/')
(train_data, train_fine_labels), (eval_data, eval_fine_labels) = get_data.load_data(label_mode='fine', path='./Data/cifar-100/')
train_fine_labels = train_fine_labels.reshape(train_fine_labels.shape[0])
eval_fine_labels = eval_fine_labels.reshape(eval_fine_labels.shape[0])
train_coarse_labels = train_coarse_labels.reshape(train_coarse_labels.shape[0])
eval_coarse_labels = eval_coarse_labels.reshape(eval_coarse_labels.shape[0])

classes = get_data.unpickle('./Data/cifar-100/meta')
fine_class = np.asarray(classes[b'fine_label_names'], dtype=np.dtype(np.str))
course_class = np.asarray(classes[b'coarse_label_names'], dtype=np.dtype(np.str))

for idx in range(train_data.shape[0]):
    img = train_data[idx].reshape((1,) + train_data[idx].shape)
    i = 0
    for batch in datagen.flow(img, batch_size=1, save_to_dir='./Data/cifar-100/augmentation', 
                                save_prefix=str(idx)+'_'+course_class[train_coarse_labels[idx]]+'_'+fine_class[train_fine_labels[idx]], save_format='jpg'):
        i += 1
        if i == 1:
            break

# img = load_img('/Desktop/lena.jpg')  # this is a PIL image, please replace to your own file path
# x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
# x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory

# i = 0
# for batch in datagen.flow(x, batch_size=1, save_to_dir='~/Desktop/preview', save_prefix='lena', save_format='jpg'): 
#    i += 1
#    if i > 20:
#        break  # otherwise the generator would loop indefinitely
