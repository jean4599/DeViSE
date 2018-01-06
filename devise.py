
# coding: utf-8

# In[2]:


import tensorflow as tf
import numpy as np
import os
from Word2Vec import Word2Vec_Model
from IPython.display import clear_output, Image, display, HTML

tf.logging.set_verbosity(tf.logging.INFO)

########## Hyperparameter ##########
BATCH_SIZE = 20
EPOCH_BOUND = 1000
EARLY_STOP_CHECK_EPOCH = 20
TAKE_CROSS_VALIDATION = False
CROSS_VALIDATION = 5
TEXT_EMBEDDING_SIZE = 50
MARGIN = 0.1
STORED_PATH = "./devise_model/devise.ckpt"
########## Hyperparameter ##########

########## load Word2Vec model ##########
text_embedding_model = Word2Vec_Model(word2vec_model_path="./Data/glove.6B/glove.6B.50d.txt")
all_text_embedding = text_embedding_model.load_light_word2vec()
W2V_texts = np.array(list(all_text_embedding.keys()), dtype=np.str)
print('W2V_texts', W2V_texts.shape)
print('ScikitLearn Nearest Neighbors: ', text_embedding_model.train_nearest_neighbor(all_text_embedding, num_nearest_neighbor=5))
    
########## load Word2Vec model ##########

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

def devise_model(features, labels, mode):
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
        training= mode=='TRAIN',
        name='dropout1')
    dense2 = tf.layers.dense(
        inputs=dropout1,
        units=512, # number of neurons in the dense layer
        activation=tf.nn.relu,
        name='dense2')
    dropout2 = tf.layers.dropout(
        inputs=dense2,
        rate=0.1,
        training= mode=='TRAIN',
        name='dropout2')
    ########## Core Visual Model ##########
    
    ########## Transformation ##########
    visual_embeddings = tf.layers.dense(inputs=dropout2, units=TEXT_EMBEDDING_SIZE, name='transform')
    tf.summary.histogram('visual_embeddings', visual_embeddings)
    ########## Transformation ##########
    
    return visual_embeddings

def max_margin_loss(visual_embeddings, y):
    global classes_text_embedding, MARGIN, BATCH_SIZE
    
#     loss = 0.0
#     for i in range(BATCH_SIZE):
#         for j in range(len(classes_text_embedding)):
#             loss += tf.maximum(0.0, (MARGIN - tf.tensordot(y[i], visual_embeddings[i], axes=1) 
#                             + tf.tensordot(classes_text_embedding[j], visual_embeddings[i], axes=1)))
#         loss -= MARGIN
#     loss /= BATCH_SIZE
    with tf.name_scope('loss'):
        loss = tf.constant(0.0)

        predic_true_distance = tf.reduce_sum(tf.multiply(y, visual_embeddings), axis=1, keep_dims=True)
        print("predic_true_distance:", predic_true_distance)
        for j in range(len(classes_text_embedding)):
            loss = tf.add(loss, tf.maximum(0.0, (MARGIN - predic_true_distance 
                                    + tf.reduce_sum(tf.multiply(classes_text_embedding[j], visual_embeddings),axis=1, keep_dims=True))))
        loss = tf.subtract(loss, MARGIN)
        loss = tf.reduce_sum(loss)
        loss = tf.div(loss, BATCH_SIZE)

#     num_classes = len(CLASSES)
#     loss = 0.0
#     t_labelMv = (-num_classes) * tf.reduce_sum(tf.multiply(y, visual_embeddings))
#     for i, class_text_embedding in enumerate (classes_text_embedding):
#         t_jMv = tf.reduce_sum(tf.multiply(class_text_embedding, visual_embeddings))
#         loss += t_jMv
#     loss += t_labelMv
#     loss = tf.maximum(0.0, MARGIN*(num_classes-1) + loss) / BATCH_SIZE
#     print("calculate loss...")
#     print('visual embedding: ', visual_embeddings)
#     num_classes = len(CLASSES)
#     # t_labelMv
#     loss = (-num_classes) * tf.multiply(y, visual_embeddings)
#     # t_jMv
#     for i, class_text_embedding in enumerate (classes_text_embedding):
#         loss += tf.multiply(class_text_embedding, visual_embeddings)

#     print('tf.reduce_sum(loss, axis=1, keep_dims=True):', tf.reduce_sum(loss, axis=1, keep_dims=True))
#     loss = tf.reduce_sum(tf.maximum(0.0, (MARGIN*(num_classes-1) + tf.reduce_sum(loss, axis=1, keep_dims=True)))) / BATCH_SIZE
#     print('visual embedding: ', visual_embeddings)
    
    return loss


def train(X_train, y_train, X_validate, y_validate, optimizer, epoch_bound, stop_threshold, batch_size, testing=False):

    global saver
    global loss ,writer, merged
    
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
                                               mode:'TRAIN'})
            else:
                sess.run(optimizer, feed_dict={x: X_train[batch*batch_size : (batch+1)*batch_size], 
                                               y: y_train[batch*batch_size : (batch+1)*batch_size], 
                                               mode:'TRAIN'})
        # print('Validating...')
       
        # split validation set into multiple mini-batches and start validating
        cur_loss = 0.0
        total_batches = int(X_validate.shape[0] / batch_size)
        for batch in range(total_batches):
            # cur_loss = tf.add(cur_loss, loss)
            # tf.summary.scalar('loss', cur_loss)
            
            #print('Merged: ', merged)
            if batch == total_batches - 1:
                cur_loss += sess.run(loss, feed_dict={x:X_validate[batch*batch_size:],
                                                     y:y_validate[batch*batch_size:],
                                                     mode:'EVAL'})
            else:
                cur_loss += sess.run(loss, feed_dict={x:X_validate[batch*batch_size : (batch+1)*batch_size],
                                                     y:y_validate[batch*batch_size : (batch+1)*batch_size],
                                                     mode:'EVAL'})
        cur_loss /= total_batches

        #writer.add_summary(summary, tf.train.get_global_step())
        
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
    
    global text_embedding_model, all_text_embedding, TEXT_EMBEDDING_SIZE, CLASSES
    
    labels_embeddings = []
    for i in labels:
        labels_embeddings.append(text_embedding_model.text_embedding_lookup(all_text_embedding, TEXT_EMBEDDING_SIZE, CLASSES[i]))
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


########## Data ##########
# Data format:
# data -- a 10000x3072 numpy array of uint8s. 
#         Each row of the array stores a 32x32 colour image. 
#         The first 1024 entries contain the red channel values, 
#         the next 1024 the green, and the final 1024 the blue. 
#         The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.
# labels -- a list of 10000 numbers in the range 0-99. The number at index i indicates the label of the ith image in the array data.

# Load training data
train_set = unpickle('./Data/cifar-100/train')
train_data = np.asarray(train_set[b'data'], dtype=np.float32) # shape (50000, 3072) 50000 images of 32x32x3 values
train_labels = np.asarray(train_set[b'fine_labels'], dtype=np.int32)

# Load testing data
test_set = unpickle('./Data/cifar-100/test')
eval_data = np.asarray(test_set[b'data'], dtype=np.float32) # shape (10000, 3072) 50000 images of 32x32x3 values
eval_labels = np.asarray(test_set[b'fine_labels'], dtype=np.int32)

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
print("train_labels_embeddings shape:", train_labels_embeddings.shape, "type:", train_labels_embeddings.dtype)
print("eval_labels_embeddings shape:", eval_labels_embeddings.shape, "type:", eval_labels_embeddings.dtype)


classes_text_embedding = text_embedding_model.get_classes_text_embedding(all_text_embedding, TEXT_EMBEDDING_SIZE, CLASSES)
print('classes_text_embedding shape:', classes_text_embedding.shape)
print('classes_text_embedding len:', len(classes_text_embedding))
#nearest_neighbors = text_embedding_model.nearest_neighbor_embeddings(classes_text_embedding[2], all_text_embedding, 10)
#print("Nearest neighbors", nearest_neighbors[0][0])

print('Train Data shape: ',train_data.shape)
print('Train Label shape: ', train_labels.shape)
#print(all_text_embedding['baby', 'sister'])

########## Data ##########

########## devise classifier ##########
x = tf.placeholder(tf.float32, [None, train_data.shape[1]], name='x')
y = tf.placeholder(tf.float32, [None, train_labels_embeddings.shape[1]], name='y')
mode = tf.placeholder(tf.string, name='mode')

visual_embeddings = devise_model(x, y, mode)
print('visual_embeddings:', visual_embeddings)
print('y', y)
# Calculate Loss (for both TRAIN and EVAL modes)
#onehot_labels = tf.one_hot(indices=tf.cast(y, tf.int32), depth=100)
#loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
# visual_embeddings = np.ones((BATCH_SIZE, 300), dtype=np.float32)
loss = max_margin_loss(visual_embeddings, y)

print("loss defined")
# Training iteration (for TRAIN)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())

# # For prediction (for EVAL)
# nearest_neighbor_labels = tf.convert_to_tensor([], tf.string)
# for batch in range(int(train_data.shape[0]/BATCH_SIZE)):
#     labels = tf.py_func(func=text_embedding_model.get_nearest_neighbor_labels,
#                         inp=[visual_embeddings[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE], W2V_texts],
#                         Tout=[tf.string]) # get 5 nearest neighbors of text_embedding for each visual_embedding
#     labels = tf.convert_to_tensor(labels, tf.string)
#     tf.concat([nearest_neighbor_labels, labels], axis=0)

# predictions = {
#     # Generate predictions (for PREDICT and EVAL mode)
#     "visual_embeddings": visual_embeddings,
#     "nearest_neighbors": nearest_neighbor_labels,
# }

#show_graph(tf.get_default_graph().as_graph_def())

########## devise classifier ##########

########## Train ##########
print("########## Start training ##########")
sess = tf.Session()
init = tf.global_variables_initializer()
# init saver to save model
saver = tf.train.Saver()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("logs/", sess.graph)

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


# if os.path.exists(STORED_PATH+".meta") == True:
#     # restore the precious best model
#     saver.restore(sess, STORED_PATH)
# else:
#     # init weights
#     sess.run(init)

# init weights
sess.run(init)
    
# randomize dataset
indices = np.random.permutation(train_data.shape[0])
Inputs, Labels = np.array(train_data[indices,:]), np.array(train_labels_embeddings[indices,:])

# get validation set with the size of a batch for early-stop
X_train, y_train = Inputs[BATCH_SIZE:], Labels[BATCH_SIZE:]
X_validate, y_validate = Inputs[:BATCH_SIZE], Labels[:BATCH_SIZE]

# start training with all the inputs
best_loss = train(X_train, y_train, X_validate, y_validate
                        , train_op, EPOCH_BOUND, EARLY_STOP_CHECK_EPOCH, BATCH_SIZE, testing=True)
print("training with all the inputs, loss:", best_loss)


sess.close()
########## Train ##########


########## Evaluate ##########
# Evaluate the model and print results
print("########## Start evaluating ##########")
sess = tf.Session()
# restore the precious best model
saver.restore(sess, STORED_PATH)

# predictions = text_embedding_model.get_nearest_neighbor_labels(visual_embeddings, W2V_texts)
# #testing_accuracy = 0.0
# for i in range(5):
#     results = sess.run(predictions, feed_dict={x:eval_data[i*2000:(i+1)*2000], y:eval_labels_embeddings[i*2000:(i+1)*2000], mode:'EVAL'})
#     #testing_accuracy += sess.run(tf.reduce_mean(tf.cast(tf.equal(eval_labels[i*2000:(i+1)*2000], results['classes']), tf.float32)))
#     print('Predic nearest neighbor: ', results['nearest_neighbors'])

#testing_accuracy = 0.0
for i in range(10):
    predict_embeddings = sess.run(visual_embeddings, feed_dict={x:eval_data[i*200:(i+1)*200], y:eval_labels_embeddings[i*200:(i+1)*200], mode:'EVAL'})
    predict_labels = text_embedding_model.get_nearest_neighbor_labels(predict_embeddings, W2V_texts)
    print('Predic nearest neighbor: ')
    for idx, label in enumerate(predict_labels):
        print('Predict top 5 labels:', label, 'True lable:', eval_labels[idx])

#print('Test accuracy: ', testing_accuracy/5)
sess.close()
######### Evaluate ##########



