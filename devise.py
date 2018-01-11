
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import os
import get_data
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
########## Hyperparameter ##########

########## load Word2Vec model ##########
# TextEmbeddings = TextEmbeddings(word2vec_model_path="./Data/glove.6B/glove.6B.50d.txt")
# all_text_embedding = TextEmbeddings.load_light_word2vec()
# W2V_texts = np.array(list(all_text_embedding.keys()), dtype=np.str)
# print('W2V_texts', W2V_texts.shape)

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


# In[2]:


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
        
def labels_2_embeddings(labels):
    
    global TextEmbeddings, TEXT_EMBEDDING_SIZE, CLASSES
    
    labels_embeddings = []
    for i in labels:
        labels_embeddings.append(TextEmbeddings.text_embedding_lookup(TEXT_EMBEDDING_SIZE, CLASSES[i]))
    labels_embeddings = np.array(labels_embeddings, dtype=np.float32)
    
    return labels_embeddings
    

# split dataset into training set and one validation set
def split_folds(indices, Inputs, Labels, cross_validation, fold):
    n = Inputs.shape[0]
    if fold == cross_validation:
        validation_size = n - (int(n/cross_validation) * (cross_validation-1))
        X_train_idx, X_validate_idx = indices[:(n-validation_size)], indices[(n-validation_size):]
        y_train_idx, y_validate_idx = indices[:(n-validation_size)], indices[(n-validation_size):]
    else:
        validation_size = int(n/cross_validation)
        X_train_idx, X_validate_idx = np.concatenate((indices[:validation_size*(fold-1)], indices[validation_size*fold:]), axis=0), indices[(validation_size*(fold-1)):(validation_size*fold)]
        y_train_idx, y_validate_idx = np.concatenate((indices[:validation_size*(fold-1)], indices[validation_size*fold:]), axis=0), indices[(validation_size*(fold-1)):(validation_size*fold)]
    X_train, X_validate = np.array(Inputs[X_train_idx,:]), np.array(Inputs[X_validate_idx,:])
    y_train, y_validate = np.array(Labels[y_train_idx,:]), np.array(Labels[y_validate_idx,:])
    return X_train, y_train, X_validate, y_validate


# In[3]:


import data_augmentation
########## Data ##########
# Data format:
# data -- a 10000x3072 numpy array of uint8s. 
#         Each row of the array stores a 32x32 colour image. 
#         The first 1024 entries contain the red channel values, 
#         the next 1024 the green, and the final 1024 the blue. 
#         The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.
# labels -- a list of 10000 numbers in the range 0-99. The number at index i indicates the label of the ith image in the array data.

# Load training and testing data
# (train_data, train_labels), (eval_data, eval_labels) = get_data.load_data(path='./Data/cifar-100/')
# train_data = train_data.reshape(train_data.shape[0], 32*32*3)
# train_labels = train_labels.reshape(train_labels.shape[0])
# eval_data = eval_data.reshape(eval_data.shape[0], 32*32*3)
# eval_labels = eval_labels.reshape(eval_labels.shape[0])

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
# print('ScikitLearn Nearest Neighbors: ', Word2Vec_Model.train_nearest_neighbor(classes_text_embedding, num_nearest_neighbor=5))

print('Train Data shape: ',train_data.shape)
print('Train Label shape: ', train_labels.shape)
########## Data ##########


# In[4]:


########## devise classifier ##########
# x = tf.placeholder(tf.float32, [None, train_data.shape[1]], name='x')

# mode = tf.placeholder(tf.string, name='mode')

# visual_embeddings = devise_model(x, y, mode)

reset_graph()
# Transfer layers
pretrained_model_path = "./saved_model/cifar100_simpleCNN/cifar-100_simpleCNN"
# Get graph from pretrained visual model
pretrained_saver = tf.train.import_meta_graph(pretrained_model_path + ".ckpt.meta")
# get variables of cifar-100 cnn model
x = tf.get_default_graph().get_tensor_by_name("x:0")
y = tf.placeholder(tf.float32, [None, train_labels_embeddings.shape[1]], name='y')
yy = tf.get_default_graph().get_tensor_by_name("y:0")
train_mode = tf.get_default_graph().get_tensor_by_name("train_mode:0")
# print(tf.get_default_graph().get_operations())
cnn_output = tf.get_default_graph().get_tensor_by_name("dropout1/cond/Merge:0")
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
writer = tf.summary.FileWriter("logs/cifar100_simpleCNN_devise", sess.graph)

# init weights
sess.run(init)
# restore value from pretrained model
pretrained_saver.restore = (sess, pretrained_model_path + ".ckpt")
for var in training_vars:
    sess.run(var.initializer)
# randomize dataset
indices = np.random.permutation(train_data.shape[0])
# start cross validation
avg_loss = 0.0

if TAKE_CROSS_VALIDATION == True:
    for fold in range(1, CROSS_VALIDATION+1):
        print("########## Fold:", fold, "##########")
        if os.path.exists(STORED_PATH+".meta") == True:
            # restore the precious best model
            saver.restore(sess, STORED_PATH)
        else:
            # init weights
            sess.run(init)
            
        # split inputs into training set and validation set for each fold
        X_train, y_train, X_validate, y_validate = split_folds(indices, train_data, train_labels_embeddings, CROSS_VALIDATION, fold)
        print('validate data: ', X_validate.shape)
        print('validate label: ', y_validate.shape)
        print('train data: ', X_train.shape)
        print('train label: ', y_train.shape)

        best_loss = train(X_train, y_train, X_validate, y_validate
                                , train_op, EPOCH_BOUND, EARLY_STOP_CHECK_EPOCH, BATCH_SIZE, testing=False)
        avg_loss += best_loss
        print("Loss:", best_loss)
    avg_loss /= cross_validation
    print("Average loss of cross validation:", avg_loss)
    
# randomize dataset
indices = np.random.permutation(train_data.shape[0])
Inputs, Labels = np.array(train_data[indices,:]), np.array(train_labels_embeddings[indices,:])
yy_Labels = np.array(train_labels[indices])

# get validation set with the size of a batch for early-stop
X_train, y_train = Inputs[VALIDATION_SIZE:], Labels[VALIDATION_SIZE:]
X_validate, y_validate = Inputs[:VALIDATION_SIZE], Labels[:VALIDATION_SIZE]

yy_train = yy_Labels[VALIDATION_SIZE:]
yy_validation = yy_Labels[:VALIDATION_SIZE]

print('X_train[0]: ', X_train[0])

# start training with all the inputs
best_loss = train(X_train, y_train, yy_train, X_validate, y_validate, yy_validation,
                        train_op, EPOCH_BOUND, EARLY_STOP_CHECK_EPOCH, BATCH_SIZE, testing=True)
print("training with all the inputs, loss:", best_loss)


sess.close()
########## Train ##########


# In[6]:


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
                                                                y:eval_labels_embeddings[i*1000:(i+1)*1000],
                                                                yy: eval_labels[i*1000:(i+1)*1000],
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

