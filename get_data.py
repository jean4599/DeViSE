import os
import sys
import time
import random
import numpy as np
from six.moves import cPickle

class_num = 100
image_size = 32
img_channels = 3

########## Data ##########
# Data format:
# data -- a 10000x3072 numpy array of uint8s. 
#         Each row of the array stores a 32x32 colour image. 
#         The first 1024 entries contain the red channel values, 
#         the next 1024 the green, and the final 1024 the blue. 
#         The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.
# labels -- a list of 10000 numbers in the range 0-99. The number at index i indicates the label of the ith image in the array data.

# # Load training data
# train_set = unpickle('./Data/cifar-100/train')
# train_data = np.asarray(train_set[b'data'], dtype=np.float32) # shape (50000, 3072) 50000 images of 32x32x3 values
# train_labels = np.asarray(train_set[b'fine_labels'], dtype=np.int32)
# filenames = np.asarray(train_set[b'filenames'])


# ========================================================== #
# ├─ prepare_data()
#  ├─ download training data if not exist by download_data()
#  ├─ load data by load_data()
#  └─ shuffe and return data
# ========================================================== #



def load_batch(fpath):
  """Internal utility for parsing CIFAR data.
  Arguments:
      fpath: path the file to parse.
      label_key: key for label data in the retrieve
          dictionary.
  Returns:
      A tuple `(data, labels)`.
  """
  f = open(fpath, 'rb')
  if sys.version_info < (3,):
    d = cPickle.load(f)
  else:
    d = cPickle.load(f, encoding='bytes')
    # decode utf8
    d_decoded = {}
    for k, v in d.items():
      d_decoded[k.decode('utf8')] = v
    d = d_decoded
  f.close()
  data = d['data']
  fine_labels = d['fine_labels']
  coarse_labels = d['coarse_labels']

  data = data.reshape(data.shape[0], 3, 32, 32)
  return data, fine_labels, coarse_labels


def load_data(label_mode='fine', path='./Data/cifar-100'):
  """Loads CIFAR100 dataset.
  Arguments:
      Pata path.
  Returns:
      Tuple of Numpy arrays: `(x_train, y_train_fine_label, y_train_coarse_label), (x_test, y_test_fine_label, y_test_coarse_label)`.
  Raises:
      ValueError: in case of invalid `label_mode`.
  """
  if label_mode not in ['fine', 'coarse', 'both']:
    raise ValueError('label_mode must be one of "fine", "coarse", "both"')

  fpath = os.path.join(path, 'train')
  x_train, y_train_fine_label, y_train_coarse_label = load_batch(fpath)

  fpath = os.path.join(path, 'test')
  x_test, y_test_fine_label, y_test_coarse_label = load_batch(fpath)

  y_train_fine_label = np.reshape(y_train_fine_label, (len(y_train_fine_label)))
  y_train_coarse_label = np.reshape(y_train_coarse_label, (len(y_train_coarse_label)))
  y_test_fine_label = np.reshape(y_test_fine_label, (len(y_test_fine_label)))
  y_test_coarse_label = np.reshape(y_test_coarse_label, (len(y_test_coarse_label)))

  x_train = x_train.transpose(0, 2, 3, 1)
  x_test = x_test.transpose(0, 2, 3, 1)

  if label_mode=='fine':
    return  (x_train, y_train_fine_label), (x_test, y_test_fine_label)
  elif label_mode=='coarse':
    return (x_train, y_train_coarse_label), (x_test, y_test_coarse_label)
  else:
    return (x_train, y_train_fine_label, y_train_coarse_label), (x_test, y_test_fine_label, y_test_coarse_label)


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict # return dic keys: [b'filenames', b'batch_label', b'fine_labels', b'coarse_labels', b'data']

