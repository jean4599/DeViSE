import tensorflow as tf
import numpy as np
import os
import get_data
import data_augmentation
from Word2Vec import Word2Vec_Model

# What model to be evaluated?
STORED_PATH = "./saved_model/cifar100_simpleCNN_newdevise/devise.ckpt"

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

########## load Word2Vec model ##########
TextEmbeddings = Word2Vec_Model(word2vec_model_path="./Data/wiki.en.vec")
########## Load testing Data ##########
(x_train, y_train_fine_label, y_train_coarse_label), (x_test, y_test_fine_label, y_test_coarse_label) = get_data.load_data(label_mode='both')
eval_data = eval_data.reshape(eval_data.shape[0], 32*32*3)

# normalize inputs from 0-255 to 0.0-1.0
eval_data = eval_data / 255.0

# 100 labels of cifar-100
# cifar-100 class list
# fine_labels: 100 labels of classes
# coarse_labels: 20 labels of super classes
classes = unpickle('./Data/cifar-100/meta')
FINE_CLASSES = np.asarray(classes[b'fine_label_names'], dtype=np.dtype(np.str))
COARSE_CLASSES = np.asarray(classes[b'coarse_label_names'], dtype=np.dtype(np.str))


# Get eval labels' text embedding from Word2Vec Model
eval_fine_labels_embeddings = labels_2_embeddings(y_eval_fine_labels, FINE_CLASSES)

# Get class labels' text embedding from Word2Vec Model
fine_classes_text_embedding = TextEmbeddings.get_classes_text_embedding(TEXT_EMBEDDING_SIZE, FINE_CLASSES)

print('Test Data shape: ',x_test.shape)
print('Test Fine Label shape: ', y_test_fine_label.shape)
print('Test Coarse Label shape:', y_test_coarse_label.shape)

########## Evaluate ##########
# Evaluate the model and print results
print("########## Start evaluating ##########")
sess = tf.Session()
# restore the precious best model
saver = tf.train.Saver()
saver.restore(sess, STORED_PATH)

# train nearest neighbor model based on fine classes
TextEmbeddings.train_nearest_neighbor(fine_classes_text_embedding, num_nearest_neighbor=5)

top1_hit = 0.0
top3_hit = 0.0
top5_hit = 0.0
for i in range(10):
    predict_embeddings = sess.run(visual_embeddings, feed_dict={x:x_test[i*1000:(i+1)*1000],
                                                                y1:eval_coarse_labels_embeddings[i*1000:(i+1)*1000],
                                                                y2:eval_fine_labels_embeddings[i*1000:(i+1)*1000],
                                                                yy: y_eval_fine_labels[i*1000:(i+1)*1000],
                                                                train_mode: False})
    predict_batch_labels = TextEmbeddings.get_nearest_neighbor_labels_from_definedset(predict_embeddings, 1)
    for idx, predict_labels_idices in enumerate(predict_batch_labels):
        true_label = FINE_CLASSES[y_eval_fine_labels[idx]] # refer to class labels       
        for predict_label_idx in predict_labels_idices:
        	if FINE_CLASSES[predict_label_idx] == true_label:
        		top1_hit+=1
        		print("top1 HIT!")
    predict_batch_labels = TextEmbeddings.get_nearest_neighbor_labels_from_definedset(predict_embeddings, 3)
    for idx, predict_labels_idx in enumerate(predict_batch_labels):
        true_label = FINE_CLASSES[y_eval_fine_labels[idx]] # refer to class labels       
        for predict_label_idx in predict_labels_idices:
        	if FINE_CLASSES[predict_label_idx] == true_label:
        		top3_hit+=1
        		print("top3 HIT!")
                
    predict_batch_labels = TextEmbeddings.get_nearest_neighbor_labels_from_definedset(predict_embeddings, 5)
    for idx, predict_labels_idx in enumerate(predict_batch_labels):
        true_label = FINE_CLASSES[y_eval_fine_labels[idx]] # refer to class labels       
        for predict_label_idx in predict_labels_idices:
        	if FINE_CLASSES[predict_label_idx] == true_label:
        		top5_hit+=1
        		print("top5 HIT!")


print("Test result: Top 1 hit rate", top1_hit/100, "%")
print("Test result: Top 3 hit rate", top3_hit/100, "%")
print("Test result: Top 5 hit rate", top5_hit/100, "%")

sess.close()
######### Evaluate ##########
