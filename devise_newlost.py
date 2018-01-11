
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
STORED_PATH = "./saved_model/cifar100_simpleCNN_newdevise/devise.ckpt"
########## Hyperparameter ##########

########## load Word2Vec model ##########
# TextEmbeddings = TextEmbeddings(word2vec_model_path="./Data/glove.6B/glove.6B.50d.txt")
# all_text_embedding = TextEmbeddings.load_light_word2vec()
# W2V_texts = np.array(list(all_text_embedding.keys()), dtype=np.str)
# print('W2V_texts', W2V_texts.shape)

TextEmbeddings = Word2Vec_Model(word2vec_model_path="./Data/wiki.en.vec")
    


# In[2]:


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

def weight_variable(shape, w=0.1):
    initial = tf.truncated_normal(shape, stddev=w) #Outputs random values from a truncated normal distribution.
    return tf.Variable(initial)

def bias_variable(shape, w=0.1):
    initial = tf.constant(w, shape=shape)
    return tf.Variable(initial)

def devise_model(features, labels, train_mode):
    # Input Layer
    # input layer shape should be [batch_size, image_width, image_height, channels] for conv2d
    # set batch_size = -1 means batch_size = the number of input
    print('input data shape: ', features)
    input_layer = tf.reshape(features, [-1, 32, 32, 3])
    print('input layer shape: ', input_layer.shape)
    
    ########## Core Visual Model ##########
    # conv1
    with tf.variable_scope('conv1') as scope:
        kernel = weight_variable(shape=[5, 5, 3, 64]) #shape=[filter_height * filter_width * in_channels, output_channels]
        conv = tf.nn.conv2d(input_layer, kernel, [1, 1, 1, 1], padding='SAME')
        biases = bias_variable(shape=[64], w=0.0)
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
       
    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    # norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
    # conv2
    with tf.variable_scope('conv2') as scope:
        kernel = weight_variable(shape=[5, 5, 64, 64])
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = bias_variable(shape=[64], w=0.1)
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
    # pool2    
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
    # norm2
    norm2 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    
    pool2_flat = tf.reshape(norm2, [-1, 8*8*64])
    dense1 = tf.layers.dense(
        inputs=pool2_flat,
        units=1024, # number of neurons in the dense layer
        activation=tf.nn.relu,
        name='dense1')
    dropout1 = tf.layers.dropout(
        inputs=dense1,
        rate=0.1,
        training= train_mode,
        name='dropout1')
    dense2 = tf.layers.dense(
        inputs=dropout1,
        units=512, # number of neurons in the dense layer
        activation=tf.nn.relu,
        name='dense2')
    dropout2 = tf.layers.dropout(
        inputs=dense2,
        rate=0.1,
        training= train_mode,
        name='dropout2')
    ########## Core Visual Model ##########
    
    ########## Transformation ##########
    visual_embeddings = tf.layers.dense(inputs=dropout2, units=TEXT_EMBEDDING_SIZE, name='transform')
    tf.summary.histogram('visual_embeddings', visual_embeddings)
    ########## Transformation ##########
    
    return visual_embeddings


# In[3]:


def train(X_train, y_fine_train, y_coarse_train, yy_train, X_validate, y_fine_validate, y_coarse_validate, yy_validate
                        , train_op, epoch_bound, stop_threshold, batch_size, testing=True):

    global saver, loss
    global writer, merged
    
    early_stop = 0
    best_loss = np.infty
    
    for epoch in range(epoch_bound):

        # randomize training set
        indices_training = np.random.permutation(X_train.shape[0])
        X_train, y_fine_train, y_coarse_train = X_train[indices_training,:], y_fine_train[indices_training,:], y_coarse_train[indices_training,:]
        yy_train = yy_train[indices_training]
        
        # split training set into multiple mini-batches and start training        
        total_batches = int(X_train.shape[0] / batch_size)
        for batch in range(total_batches):
            if batch == total_batches - 1:
                sess.run(train_op, feed_dict={x: X_train[batch*batch_size:],
                                              y1: y_coarse_train[batch*batch_size:],
                                              y2: y_fine_train[batch*batch_size:],
                                              yy: yy_train[batch*batch_size:],
                                               train_mode: True})
                summary = sess.run(merged, feed_dict={x: X_train[batch*batch_size:],
                                                      y1: y_coarse_train[batch*batch_size:],
                                                      y2: y_fine_train[batch*batch_size:],
                                                      yy: yy_train[batch*batch_size:],
                                                      train_mode: True})
                writer.add_summary(summary, epoch + (batch/total_batches))


            else:
                sess.run(train_op, feed_dict={x: X_train[batch*batch_size : (batch+1)*batch_size], 
                                               y1: y_coarse_train[batch*batch_size : (batch+1)*batch_size],
                                               y2: y_fine_train[batch*batch_size : (batch+1)*batch_size],
                                               yy: yy_train[batch*batch_size : (batch+1)*batch_size],
                                               train_mode: True})
                summary = sess.run(merged, feed_dict={x: X_train[batch*batch_size : (batch+1)*batch_size],
                                                      y1: y_coarse_train[batch*batch_size : (batch+1)*batch_size],
                                                      y2: y_fine_train[batch*batch_size : (batch+1)*batch_size],
                                                      yy: yy_train[batch*batch_size : (batch+1)*batch_size],
                                                      train_mode: True})
                writer.add_summary(summary, epoch + (batch/total_batches))

        # split validation set into multiple mini-batches and start validating
        cur_loss = 0.0
        total_batches = int(X_validate.shape[0] / batch_size)
        for batch in range(total_batches):
            
            if batch == total_batches - 1:
                cur_loss += sess.run(loss, feed_dict={x:X_validate[batch*batch_size:],
                                                      y1:y_coarse_validate[batch*batch_size:],
                                                      y2:y_fine_validate[batch*batch_size:],
                                                      yy: yy_validate[batch*batch_size:],
                                                      train_mode: False})
            else:
                cur_loss += sess.run(loss, feed_dict={x:X_validate[batch*batch_size : (batch+1)*batch_size],
                                                      y1:y_coarse_validate[batch*batch_size : (batch+1)*batch_size],
                                                      y2:y_fine_validate[batch*batch_size : (batch+1)*batch_size],
                                                      yy: yy_validate[batch*batch_size : (batch+1)*batch_size],
                                                      train_mode: False})
        cur_loss /= total_batches
        
        # #test for prediction
        # prediction = sess.run(predictions, feed_dict={x:X_validate, y:y_validate, mode:'EVAL'})
        # print('Predic nearest neighbor: ', prediction['nearest_neighbors'])
        
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


# In[4]:


def load_batch(fpath):
    import sys
    from six.moves import cPickle
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


# In[5]:


########## Data ##########
# Data format:
# data -- a 10000x3072 numpy array of uint8s. 
#         Each row of the array stores a 32x32 colour image. 
#         The first 1024 entries contain the red channel values, 
#         the next 1024 the green, and the final 1024 the blue. 
#         The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.
# labels -- a list of 10000 numbers in the range 0-99. The number at index i indicates the label of the ith image in the array data.

# Load training and testing data
# load cifar data and data augmentation
(train_data, y_train_fine_labels, y_train_coarse_labels), (eval_data, y_eval_fine_labels, y_eval_coarse_labels) = data_augmentation.generate(label_mode='both')
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
FINE_CLASSES = np.asarray(classes[b'fine_label_names'], dtype=np.dtype(np.str))
COARSE_CLASSES = np.asarray(classes[b'coarse_label_names'], dtype=np.dtype(np.str))
COARSE_CLASSES[-1]='vehicles'
COARSE_CLASSES[-2]='vehicles'
COARSE_CLASSES[9]='large_outdoor_things'
COARSE_CLASSES[-7]='invertebrates'

print(COARSE_CLASSES)
train_fine_labels_embeddings = labels_2_embeddings(y_train_fine_labels, FINE_CLASSES)
train_coarse_labels_embeddings = labels_2_embeddings(y_train_coarse_labels, COARSE_CLASSES)

eval_fine_labels_embeddings = labels_2_embeddings(y_eval_fine_labels, FINE_CLASSES)
eval_coarse_labels_embeddings = labels_2_embeddings(y_eval_coarse_labels, COARSE_CLASSES)

fine_classes_text_embedding = TextEmbeddings.get_classes_text_embedding(TEXT_EMBEDDING_SIZE, FINE_CLASSES)
coarse_classes_text_embedding = TextEmbeddings.get_classes_text_embedding(TEXT_EMBEDDING_SIZE, COARSE_CLASSES)

print('Train Data shape: ',train_data.shape)
print('Train Fine Label shape: ', y_train_fine_labels.shape)
print('Train Coarse Label shape:', y_train_coarse_labels.shape)
########## Data ##########


# In[12]:


########## devise classifier ##########
# x = tf.placeholder(tf.float32, [None, train_data.shape[1]], name='x')

# mode = tf.placeholder(tf.string, name='mode')

# visual_embeddings = devise_model(x, y, mode)

reset_graph()
# Transfer layers
pretrained_model_path = "./saved_model/cifar100_simpleCNN/cifar-100_simpleCNN"
# Get graph from pretrained visual model
pretrained_saver = tf.train.import_meta_graph(pretrained_model_path + ".ckpt.meta")
#print(tf.get_default_graph().get_operations())

# Get variables of cifar-100 cnn model
x = tf.get_default_graph().get_tensor_by_name("x:0")
yy = tf.get_default_graph().get_tensor_by_name("y:0")
train_mode = tf.get_default_graph().get_tensor_by_name("train_mode:0")
cnn_output = tf.get_default_graph().get_tensor_by_name("dropout1/cond/Merge:0")

# Define new input label placeholder: y1 for super class labels, y2 for fine class labels
y1 = tf.placeholder(tf.float32, [None, train_coarse_labels_embeddings.shape[1]], name='y1')
y2 = tf.placeholder(tf.float32, [None, train_fine_labels_embeddings.shape[1]], name='y2')

tf.summary.histogram('cnn_output', cnn_output)

# attach transform layer
with tf.name_scope('transform'):
    visual_embeddings = tf.layers.dense(inputs=cnn_output, units=TEXT_EMBEDDING_SIZE, name='transform')
    tf.summary.histogram('visual_embeddings', visual_embeddings)

# get training parameter in transform layer for training operation
training_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="transform")

# Calculate Loss (for both TRAIN and EVAL modes)
## Origine DeViSE defined hinge rank loss
# with tf.name_scope('devise_loss'):
#     loss = tf.constant(0.0)
#     predic_true_distance = tf.reduce_sum(tf.multiply(y, visual_embeddings), axis=1, keep_dims=True)
#     for j in range(len(classes_text_embedding)):
#         loss = tf.add(loss, tf.maximum(0.0, (MARGIN - predic_true_distance 
#                                     + tf.reduce_sum(tf.multiply(classes_text_embedding[j], visual_embeddings),axis=1, keep_dims=True))))
#     loss = tf.subtract(loss, MARGIN)
#     loss = tf.reduce_sum(loss)
#     loss = tf.div(loss, BATCH_SIZE)
#     tf.summary.scalar('loss', loss)

## New loss: Hierachicical hinge rank loss
with tf.name_scope('devise_loss'):
    
    loss = tf.constant(0.0)
    ### (H1) Coarse labels hinge rank loss
    predic_true_distance = tf.reduce_sum(tf.multiply(y1, visual_embeddings), axis=1, keep_dims=True)
    h1_loss = tf.constant(0.0)
    for j in range(len(coarse_classes_text_embedding)):
        h1_loss = tf.add(h1_loss, tf.maximum(0.0, (MARGIN - predic_true_distance 
                                    + tf.reduce_sum(tf.multiply(coarse_classes_text_embedding[j], visual_embeddings),axis=1, keep_dims=True))))
    h1_loss = tf.subtract(h1_loss, MARGIN)
    h1_loss = tf.reduce_sum(h1_loss)
    h1_loss = tf.div(h1_loss, BATCH_SIZE)
    
    ### (H2) Fine labels hinge rank loss
    predic_true_distance = tf.reduce_sum(tf.multiply(y2, visual_embeddings), axis=1, keep_dims=True)
    h2_loss = tf.constant(0.0)
    for j in range(len(fine_classes_text_embedding)):
        h2_loss = tf.add(h2_loss, tf.maximum(0.0, (MARGIN - predic_true_distance 
                                    + tf.reduce_sum(tf.multiply(fine_classes_text_embedding[j], visual_embeddings),axis=1, keep_dims=True))))
    h2_loss = tf.subtract(h2_loss, MARGIN)
    h2_loss = tf.reduce_sum(h2_loss)
    h2_loss = tf.div(h2_loss, BATCH_SIZE)
    
    ### loss = alpha*H1 + beta*H2
    loss = 0.5*h1_loss + 0.5*h2_loss
    
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


# In[ ]:


########## Train ##########
print("########## Start training ##########")
sess = tf.Session()
init = tf.global_variables_initializer()
# init saver to save model
saver = tf.train.Saver()
# visualize data
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("logs/cifar100_simpleCNN_newdevise", sess.graph)

# init weights
sess.run(init)
# restore value from pretrained model
pretrained_saver.restore = (sess, pretrained_model_path + ".ckpt")
for var in training_vars:
    sess.run(var.initializer)
    
# randomize dataset
indices = np.random.permutation(train_data.shape[0])
Inputs = np.array(train_data[indices,:])
Fine_Labels = np.array(train_fine_labels_embeddings[indices,:])
Coarse_Labels = np.array(train_coarse_labels_embeddings[indices,:])
yy_Labels = np.array(y_train_fine_labels[indices])

# get validation set with the size of a batch for early-stop
X_train, y_fine_train, y_coarse_train = Inputs[VALIDATION_SIZE:], Fine_Labels[VALIDATION_SIZE:], Coarse_Labels[VALIDATION_SIZE:]
X_validate, y_fine_validate, y_coarse_validate = Inputs[:VALIDATION_SIZE], Fine_Labels[:VALIDATION_SIZE], Coarse_Labels[:VALIDATION_SIZE]

yy_train = yy_Labels[VALIDATION_SIZE:]
yy_validation = yy_Labels[:VALIDATION_SIZE]
print('yy_train', yy_train.shape)
print('yy_valid', yy_validation.shape)


# start training with all the inputs
best_loss = train(X_train, y_fine_train, y_coarse_train, yy_train, X_validate, y_fine_validate, y_coarse_validate, yy_validation,
                        train_op, EPOCH_BOUND, EARLY_STOP_CHECK_EPOCH, BATCH_SIZE, testing=True)
print("training with all the inputs, loss:", best_loss)

sess.close()
########## Train ##########


# In[ ]:


########## Evaluate ##########
# Evaluate the model and print results
print("########## Start evaluating ##########")
sess = tf.Session()
# restore the precious best model
saver = tf.train.Saver()
saver.restore(sess, STORED_PATH)

top1_hit = 0.0
top3_hit = 0.0
top5_hit = 0.0
for i in range(10):
    predict_embeddings = sess.run(visual_embeddings, feed_dict={x:eval_data[i*1000:(i+1)*1000],
                                                                y1:eval_coarse_labels_embeddings[i*1000:(i+1)*1000],
                                                                y2:eval_fine_labels_embeddings[i*1000:(i+1)*1000],
                                                                yy: y_eval_fine_labels[i*1000:(i+1)*1000],
                                                                train_mode: False})
    predict_batch_labels = TextEmbeddings.get_nearest_neighbor_labels(predict_embeddings, 1)
    for idx, predict_labels in enumerate(predict_batch_labels):
        long_true_label = CLASSES[eval_labels[idx]] # refer to class labels       
        # consider a class name is concated by multiple labels (ex., maple_tree)
        true_labels = long_true_label.split('_')
        for true in true_labels:
            if(true in predict_labels):
                top1_hit+=1
                print("top1 HIT!")
    predict_batch_labels = TextEmbeddings.get_nearest_neighbor_labels(predict_embeddings, 3)
    for idx, predict_labels in enumerate(predict_batch_labels):
        long_true_label = CLASSES[eval_labels[idx]] # refer to class labels      
        # consider a class name is concated by multiple labels (ex., maple_tree)
        true_labels = long_true_label.split('_')
        for true in true_labels:
            if(true in predict_labels):
                top3_hit+=1
                print("top3 HIT!")
    predict_batch_labels = TextEmbeddings.get_nearest_neighbor_labels(predict_embeddings, 5)
    for idx, predict_labels in enumerate(predict_batch_labels):
        long_true_label = CLASSES[eval_labels[idx]] # refer to class labels        
        # consider a class name is concated by multiple labels (ex., maple_tree)
        true_labels = long_true_label.split('_')
        for true in true_labels:
            if(true in predict_labels):
                top5_hit+=1
                print("top5 HIT!")

print("Test result: Top 1 hit rate", top1_hit/100, "%")
print("Test result: Top 3 hit rate", top3_hit/100, "%")
print("Test result: Top 5 hit rate", top5_hit/100, "%")

#print('Test accuracy: ', testing_accuracy/5)
sess.close()
######### Evaluate ##########

