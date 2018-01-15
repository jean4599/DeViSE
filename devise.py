
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import os
import get_data
import data_augmentation
from Word2Vec import Word2Vec_Model
from IPython.display import clear_output, Image, display, HTML

tf.logging.set_verbosity(tf.logging.INFO)

########## Hyperparameter ##########
BATCH_SIZE = 20
VALIDATION_SIZE = 500
EPOCH_BOUND = 1000
EARLY_STOP_CHECK_EPOCH = 20
TAKE_CROSS_VALIDATION = False
CROSS_VALIDATION = 5
TEXT_EMBEDDING_SIZE = 300
MARGIN = 0.1
EPOCH_PER_DECAY = 10
STORED_PATH = "./saved_model/cifar100_simpleCNN_devise/devise.ckpt"
PRETRAINED_MODEL_PATH = "./saved_model/cifar100_VGG/cifar-100_VGG"
LOGS_PATH = "./logs/cifar100_VGG_newdevise"
########## Hyperparameter ##########

########## load Word2Vec model ##########

TextEmbeddings = Word2Vec_Model(word2vec_model_path="./Data/wiki.en.vec")
    
########## load Word2Vec model ##########
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = graph_def
    #strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))


def train(X_train, y_train, yy_train, X_validate, y_validate, yy_validate, optimizer, epoch_bound, stop_threshold, batch_size, testing=False):

    global saver, loss
    global writer, merged
    
    early_stop = 0
    best_loss = np.infty
    
    for epoch in range(epoch_bound):

        # randomize training set
        indices_training = np.random.permutation(X_train.shape[0])
        X_train, y_train = X_train[indices_training,:], y_train[indices_training,:]

        # split training set into multiple mini-batches and start training        
        total_batches = int(X_train.shape[0] / batch_size)
        for batch in range(total_batches):
            if batch == total_batches - 1:
                sess.run(optimizer, feed_dict={x: X_train[batch*batch_size:], 
                                               y: y_train[batch*batch_size:],
                                               yy: yy_train[batch*batch_size:],
                                               train_mode: True})
                summary = sess.run(merged, feed_dict={x: X_train[batch*batch_size:],
                                                      y: y_train[batch*batch_size:],
                                                      yy: yy_train[batch*batch_size:],
                                                      train_mode: True})
                writer.add_summary(summary, epoch + (batch/total_batches))


            else:
                sess.run(optimizer, feed_dict={x: X_train[batch*batch_size : (batch+1)*batch_size], 
                                               y: y_train[batch*batch_size : (batch+1)*batch_size], 
                                               yy: yy_train[batch*batch_size : (batch+1)*batch_size],
                                               train_mode: True})
                summary = sess.run(merged, feed_dict={x: X_train[batch*batch_size : (batch+1)*batch_size],
                                                      y: y_train[batch*batch_size : (batch+1)*batch_size],
                                                      yy: yy_train[batch*batch_size : (batch+1)*batch_size],
                                                      train_mode: True})
                writer.add_summary(summary, epoch + (batch/total_batches))

        # split validation set into multiple mini-batches and start validating
        cur_loss = 0.0
        total_batches = int(X_validate.shape[0] / batch_size)
        for batch in range(total_batches):
            
            if batch == total_batches - 1:
                cur_loss += sess.run(loss, feed_dict={x:X_validate[batch*batch_size:],
                                                      y:y_validate[batch*batch_size:],
                                                      yy:yy_validate[batch*batch_size:],
                                                     train_mode: False})
            else:
                cur_loss += sess.run(loss, feed_dict={x:X_validate[batch*batch_size : (batch+1)*batch_size],
                                                      y:y_validate[batch*batch_size : (batch+1)*batch_size],
                                                      yy:yy_validate[batch*batch_size : (batch+1)*batch_size],
                                                     train_mode: False})
        cur_loss /= total_batches

        
        # If the loss does not decrease for many times, it will early stop epochs-loop 
        if best_loss > cur_loss:
            early_stop = 0
            best_loss = cur_loss
            # save best model in testing phase
            if testing == True:
                save_path = saver.save(sess, STORED_PATH)
        else:
            early_stop += 1
        print('\tEpoch: ', epoch, '\tBest loss: ', best_loss, '\tCurrent loss: ', cur_loss)
        if early_stop == stop_threshold:
            break

    return best_loss

def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            return dict # return dic keys: [b'filenames', b'batch_label', b'fine_labels', b'coarse_labels', b'data']
        
def labels_2_embeddings(labels):
    
    global TextEmbeddings, TEXT_EMBEDDING_SIZE, CLASSES
    
    labels_embeddings = []
    for i in labels:
        labels_embeddings.append(TextEmbeddings.text_embedding_lookup(TEXT_EMBEDDING_SIZE, CLASSES[i]))
    labels_embeddings = np.array(labels_embeddings, dtype=np.float32)
    
    return labels_embeddings
    

########## Data ##########
# Data format:
# data -- a 10000x3072 numpy array of uint8s. 
#         Each row of the array stores a 32x32 colour image. 
#         The first 1024 entries contain the red channel values, 
#         the next 1024 the green, and the final 1024 the blue. 
#         The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.
# labels -- a list of 10000 numbers in the range 0-99. The number at index i indicates the label of the ith image in the array data.

# load cifar data and data augmentation
(train_data, train_labels), (eval_data, eval_labels) = data_augmentation.generate()
train_data = train_data.reshape(train_data.shape[0], 32*32*3)
eval_data = eval_data.reshape(eval_data.shape[0], 32*32*3)

# normalize inputs from 0-255 to 0.0-1.0
train_data = train_data / 255.0
eval_data = eval_data / 255.0

# 100 labels of cifar-100
# cifar-100 class list
# fine_labels: 100 labels of classes
# coarse_labels: 20 labels of super classes
classes = unpickle('./Data/cifar-100/meta')
fine_class = np.asarray(classes[b'fine_label_names'], dtype=np.dtype(np.str))
course_class = np.asarray(classes[b'coarse_label_names'], dtype=np.dtype(np.str))
CLASSES = fine_class

train_labels_embeddings = labels_2_embeddings(train_labels)
eval_labels_embeddings = labels_2_embeddings(eval_labels)
classes_text_embedding = TextEmbeddings.get_classes_text_embedding(TEXT_EMBEDDING_SIZE, CLASSES)

print('Train Data shape: ',train_data.shape)
print('Train Label shape: ', train_labels.shape)
########## Data ##########

########## devise classifier ##########

reset_graph()
# Transfer layers
# Get graph from pretrained visual model
pretrained_saver = tf.train.import_meta_graph(PRETRAINED_MODEL_PATH + ".ckpt.meta")
# get variables of cifar-100 cnn model
x = tf.get_default_graph().get_tensor_by_name("x:0")
y = tf.placeholder(tf.float32, [None, train_labels_embeddings.shape[1]], name='y')
yy = tf.get_default_graph().get_tensor_by_name("y:0")
train_mode = tf.get_default_graph().get_tensor_by_name("train_mode:0")
# print(tf.get_default_graph().get_operations())
cnn_output = tf.get_default_graph().get_tensor_by_name("dropout2/cond/Merge:0")
tf.summary.histogram('cnn_output', cnn_output)

# attach transform layer
with tf.name_scope('transform'):
    visual_embeddings = tf.layers.dense(inputs=cnn_output, units=TEXT_EMBEDDING_SIZE, name='transform')
    tf.summary.histogram('visual_embeddings', visual_embeddings)

# get training parameter in transform layer for training operation
training_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="transform")

# Calculate Loss (for both TRAIN and EVAL modes)
with tf.name_scope('devise_loss'):
    loss = tf.constant(0.0)
    predic_true_distance = tf.reduce_sum(tf.multiply(y, visual_embeddings), axis=1, keep_dims=True)
    for j in range(len(classes_text_embedding)):
        loss = tf.add(loss, tf.maximum(0.0, (MARGIN - predic_true_distance 
                                    + tf.reduce_sum(tf.multiply(classes_text_embedding[j], visual_embeddings),axis=1, keep_dims=True))))
    loss = tf.subtract(loss, MARGIN)
    loss = tf.reduce_sum(loss)
    loss = tf.div(loss, BATCH_SIZE)
    tf.summary.scalar('loss', loss)
print("loss defined")

# Define optimizer and Training iteration (for TRAIN)
## Decaying learning rate exponentially based on the total training step

decay_steps = int(BATCH_SIZE * EPOCH_PER_DECAY) # Define decay steps
global_step = tf.train.get_or_create_global_step()

learning_rate = tf.train.exponential_decay(
        learning_rate=0.01, #initial learning rate
        global_step=global_step,
        decay_steps=decay_steps,
        decay_rate=0.96,
        staircase=True,
        name='ExponentialDecayLearningRate'
    )
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate, name='GD_Optimizer')
train_op = optimizer.minimize(loss, name='train_op', var_list=training_vars)
# ## exponential moving average
ema = tf.train.ExponentialMovingAverage(decay=0.9999)

with tf.control_dependencies([train_op]):
      train_op = ema.apply(training_vars) # apply exponential moving average to training operation
########## devise classifier ##########


# In[5]:


########## Train ##########
print("########## Start training ##########")
sess = tf.Session()
init = tf.global_variables_initializer()
# init saver to save model
saver = tf.train.Saver()
# visualize data
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(LOGS_PATH, sess.graph)

# init weights
sess.run(init)
# restore value from pretrained model
pretrained_saver.restore = (sess, PRETRAINED_MODEL_PATH + ".ckpt")
for var in training_vars:
    sess.run(var.initializer)
# randomize dataset
indices = np.random.permutation(train_data.shape[0])

# randomize dataset
indices = np.random.permutation(train_data.shape[0])
Inputs, Labels = np.array(train_data[indices,:]), np.array(train_labels_embeddings[indices,:])
yy_Labels = np.array(train_labels[indices])

# get validation set with the size of a batch for early-stop
X_train, y_train = Inputs[VALIDATION_SIZE:], Labels[VALIDATION_SIZE:]
X_validate, y_validate = Inputs[:VALIDATION_SIZE], Labels[:VALIDATION_SIZE]
yy_train = yy_Labels[VALIDATION_SIZE:]
yy_validation = yy_Labels[:VALIDATION_SIZE]

# start training with all the inputs
best_loss = train(X_train, y_train, yy_train, X_validate, y_validate, yy_validation,
                        train_op, EPOCH_BOUND, EARLY_STOP_CHECK_EPOCH, BATCH_SIZE, testing=True)
print("training with all the inputs, loss:", best_loss)
sess.close()
########## Train ##########


# In[6]:


# ########## Evaluate ##########
# # Evaluate the model and print results
# print("########## Start evaluating ##########")
# sess = tf.Session()
# # restore the precious best model
# saver = tf.train.Saver()
# saver.restore(sess, STORED_PATH)

# top1_hit = 0.0
# top3_hit = 0.0
# top5_hit = 0.0
# for i in range(10):
#     predict_embeddings = sess.run(visual_embeddings, feed_dict={x:eval_data[i*1000:(i+1)*1000],
#                                                                 y:eval_labels_embeddings[i*1000:(i+1)*1000],
#                                                                 yy: eval_labels[i*1000:(i+1)*1000],
#                                                                 train_mode: False})
#     predict_batch_labels = TextEmbeddings.get_nearest_neighbor_labels(predict_embeddings, 1)
#     for idx, predict_labels in enumerate(predict_batch_labels):
#         long_true_label = CLASSES[eval_labels[idx]] # refer to class labels        
#         # consider a class name is concated by multiple labels (ex., maple_tree)
#         true_labels = long_true_label.split('_')
#         for true in true_labels:
#             if(true in predict_labels):
#                 top1_hit+=1
#                 print("top1 HIT!")
#     predict_batch_labels = TextEmbeddings.get_nearest_neighbor_labels(predict_embeddings, 3)
#     for idx, predict_labels in enumerate(predict_batch_labels):
#         long_true_label = CLASSES[eval_labels[idx]] # refer to class labels        
#         # consider a class name is concated by multiple labels (ex., maple_tree)
#         true_labels = long_true_label.split('_')
#         for true in true_labels:
#             if(true in predict_labels):
#                 top3_hit+=1
#                 print("top3 HIT!")
#     predict_batch_labels = TextEmbeddings.get_nearest_neighbor_labels(predict_embeddings, 5)
#     for idx, predict_labels in enumerate(predict_batch_labels):
#         long_true_label = CLASSES[eval_labels[idx]] # refer to class labels        
#         # consider a class name is concated by multiple labels (ex., maple_tree)
#         true_labels = long_true_label.split('_')
#         for true in true_labels:
#             if(true in predict_labels):
#                 top5_hit+=1
#                 print("top5 HIT!")

# print("Test result: Top 1 hit rate", top1_hit/100, "%")
# print("Test result: Top 3 hit rate", top3_hit/100, "%")
# print("Test result: Top 5 hit rate", top5_hit/100, "%")

# #print('Test accuracy: ', testing_accuracy/5)
# sess.close()
# ######### Evaluate ##########

