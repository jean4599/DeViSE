
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import os
from Word2Vec import Word2Vec_Model

########## load Word2Vec model ##########
TextEmbeddings = Word2Vec_Model(word2vec_model_path="./Data/wiki.en.vec")


# In[2]:


import get_data
import data_augmentation
TEXT_EMBEDDING_SIZE=300

def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            return dict # return dic keys: [b'filenames', b'batch_label', b'fine_labels', b'coarse_labels', b'data']
        
def labels_2_embeddings(labels_idx, ref):
    """ Get text embeddings of labels from Text embedding lookup table
        label(i) = ref[labels_idx[i]]
    Argument: 
        labels_idx : a list of indices that should refers to ref
        ref: a list of labels
    Return:
        labels_embeddings: a list of text embeddings
    """
    global TextEmbeddings, TEXT_EMBEDDING_SIZE
    
    labels_embeddings = []
    for i in labels_idx:
        labels_embeddings.append(TextEmbeddings.text_embedding_lookup(TEXT_EMBEDDING_SIZE, ref[i]))
    labels_embeddings = np.array(labels_embeddings, dtype=np.float32)
    
    return labels_embeddings

########## Load testing Data ##########
(x_train, y_train_fine_label, y_train_coarse_label), (x_test, y_test_fine_label, y_test_coarse_label) = get_data.load_data(label_mode='both')
x_test = x_test.reshape(x_test.shape[0], 32*32*3)

# normalize inputs from 0-255 to 0.0-1.0
x_test = x_test / 255.0

# 100 labels of cifar-100
# cifar-100 class list
# fine_labels: 100 labels of classes
# coarse_labels: 20 labels of super classes
classes = unpickle('./Data/cifar-100/meta')
FINE_CLASSES = np.asarray(classes[b'fine_label_names'], dtype=np.dtype(np.str))
COARSE_CLASSES = np.asarray(classes[b'coarse_label_names'], dtype=np.dtype(np.str))

COARSE_CLASSES[-1]='vehicles'
COARSE_CLASSES[-2]='vehicles'
COARSE_CLASSES[9]='large_outdoor_things'
COARSE_CLASSES[-7]='invertebrates'

# Get eval labels' text embedding from Word2Vec Model
eval_fine_labels_embeddings = labels_2_embeddings(y_test_fine_label, FINE_CLASSES)
eval_coarse_labels_embeddings = labels_2_embeddings(y_test_coarse_label, COARSE_CLASSES)

# Get class labels' text embedding from Word2Vec Model
fine_classes_text_embedding = TextEmbeddings.get_classes_text_embedding(TEXT_EMBEDDING_SIZE, FINE_CLASSES)

print('Test Data shape: ',x_test.shape)
print('Test Fine Label shape: ', y_test_fine_label.shape)
print('Test Coarse Label shape:', y_test_coarse_label.shape)


# ########## Evaluate DeViSE (unknown labels)##########
# # Evaluate the model and print results
# # What model to be evaluated?
# STORED_PATH = "./saved_model/cifar100_simpleCNN_devise/devise.ckpt"

# # reset graph
# tf.reset_default_graph()
# tf.set_random_seed(42)
# np.random.seed(42)

# print("########## Start evaluating SimpleCNN_devise ##########")
# sess = tf.Session()
# # restore the precious best model
# saver = tf.train.import_meta_graph(STORED_PATH + ".meta")
# print('operations:',tf.get_default_graph().get_operations())
# x = tf.get_default_graph().get_tensor_by_name("x:0")
# yy = tf.get_default_graph().get_tensor_by_name("y:0")
# y = tf.get_default_graph().get_tensor_by_name("y_1:0")

# train_mode = tf.get_default_graph().get_tensor_by_name("train_mode:0")
# visual_embeddings = tf.get_default_graph().get_tensor_by_name("transform/transform/BiasAdd:0")
# saver.restore(sess, STORED_PATH)

# # train nearest neighbor model based on fine classes
# TextEmbeddings.train_nearest_neighbor(fine_classes_text_embedding, num_nearest_neighbor=5)


# # In[ ]:


# top1_hit = 0.0
# top3_hit = 0.0
# top5_hit = 0.0
# for i in range(10):
#     predict_embeddings = sess.run(visual_embeddings, feed_dict={x:x_test[i*1000:(i+1)*1000],
#                                                                 y:eval_fine_labels_embeddings[i*1000:(i+1)*1000],
#                                                                 yy: y_test_fine_label[i*1000:(i+1)*1000],
#                                                                 train_mode: False})
#     predict_batch_labels = TextEmbeddings.get_nearest_neighbor_labels(predict_embeddings, 1)
#     for idx, predict_labels in enumerate(predict_batch_labels):
#         long_true_label = FINE_CLASSES[y_test_fine_label[i*1000+idx]] # refer to class labels        
#         # consider a class name is concated by multiple labels (ex., maple_tree)
#         true_labels = long_true_label.split('_')
#         for true in true_labels:
#             if(true in predict_labels):
#                 top1_hit+=1
#                 print("top1 HIT!")
#     predict_batch_labels = TextEmbeddings.get_nearest_neighbor_labels(predict_embeddings, 3)
#     for idx, predict_labels in enumerate(predict_batch_labels):
#         long_true_label = FINE_CLASSES[y_test_fine_label[i*1000+idx]] # refer to class labels        
#         # consider a class name is concated by multiple labels (ex., maple_tree)
#         true_labels = long_true_label.split('_')
#         for true in true_labels:
#             if(true in predict_labels):
#                 top3_hit+=1
#                 print("top3 HIT!")
#     predict_batch_labels = TextEmbeddings.get_nearest_neighbor_labels(predict_embeddings, 5)
#     for idx, predict_labels in enumerate(predict_batch_labels):
#         long_true_label = FINE_CLASSES[y_test_fine_label[i*1000+idx]] # refer to class labels        
#         # consider a class name is concated by multiple labels (ex., maple_tree)
#         true_labels = long_true_label.split('_')
#         for true in true_labels:
#             if(true in predict_labels):
#                 top5_hit+=1
#                 print("top5 HIT!")

# print("Evaluate SimpleCNN devise")
# print("Test result: Top 1 hit rate", top1_hit/100, "%")
# print("Test result: Top 3 hit rate", top3_hit/100, "%")
# print("Test result: Top 5 hit rate", top5_hit/100, "%")

# sess.close()

########## Evaluate DeViSE (unknown labels)##########
# Evaluate the model and print results
# What model to be evaluated?
STORED_PATH = "./saved_model/cifar100_VGG_devise/devise.ckpt"

# reset graph
tf.reset_default_graph()
tf.set_random_seed(42)
np.random.seed(42)

print("########## Start evaluating VGG_devise ##########")
sess = tf.Session()
# restore the precious best model
saver = tf.train.import_meta_graph(STORED_PATH + ".meta")
print('operations:',tf.get_default_graph().get_operations())
x = tf.get_default_graph().get_tensor_by_name("x:0")
yy = tf.get_default_graph().get_tensor_by_name("y:0")
y = tf.get_default_graph().get_tensor_by_name("y_1:0")

train_mode = tf.get_default_graph().get_tensor_by_name("train_mode:0")
visual_embeddings = tf.get_default_graph().get_tensor_by_name("transform/transform/BiasAdd:0")
saver.restore(sess, STORED_PATH)

# train nearest neighbor model based on fine classes
TextEmbeddings.train_nearest_neighbor(fine_classes_text_embedding, num_nearest_neighbor=5)


# In[ ]:


top1_hit = 0.0
top3_hit = 0.0
top5_hit = 0.0
for i in range(10):
    predict_embeddings = sess.run(visual_embeddings, feed_dict={x:x_test[i*1000:(i+1)*1000],
                                                                y:eval_fine_labels_embeddings[i*1000:(i+1)*1000],
                                                                yy: y_test_fine_label[i*1000:(i+1)*1000],
                                                                train_mode: False})
    predict_batch_labels = TextEmbeddings.get_nearest_neighbor_labels(predict_embeddings, 1)
    for idx, predict_labels in enumerate(predict_batch_labels):
        long_true_label = FINE_CLASSES[y_test_fine_label[i*1000+idx]] # refer to class labels        
        # consider a class name is concated by multiple labels (ex., maple_tree)
        true_labels = long_true_label.split('_')
        for true in true_labels:
            if(true in predict_labels):
                top1_hit+=1
                print("top1 HIT!")
    predict_batch_labels = TextEmbeddings.get_nearest_neighbor_labels(predict_embeddings, 3)
    for idx, predict_labels in enumerate(predict_batch_labels):
        long_true_label = FINE_CLASSES[y_test_fine_label[i*1000+idx]] # refer to class labels        
        # consider a class name is concated by multiple labels (ex., maple_tree)
        true_labels = long_true_label.split('_')
        for true in true_labels:
            if(true in predict_labels):
                top3_hit+=1
                print("top3 HIT!")
    predict_batch_labels = TextEmbeddings.get_nearest_neighbor_labels(predict_embeddings, 5)
    for idx, predict_labels in enumerate(predict_batch_labels):
        long_true_label = FINE_CLASSES[y_test_fine_label[i*1000+idx]] # refer to class labels        
        # consider a class name is concated by multiple labels (ex., maple_tree)
        true_labels = long_true_label.split('_')
        for true in true_labels:
            if(true in predict_labels):
                top5_hit+=1
                print("top5 HIT!")

print("Evaluate VGG devise")
print("Test result: Top 1 hit rate", top1_hit/100, "%")
print("Test result: Top 3 hit rate", top3_hit/100, "%")
print("Test result: Top 5 hit rate", top5_hit/100, "%")

sess.close()
######### Evaluate ##########


