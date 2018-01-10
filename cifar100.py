
# coding: utf-8

# In[ ]:


import tensorflow as tf
import numpy as np
import os
import get_data


tf.logging.set_verbosity(tf.logging.INFO)

########## Hyperparameter ##########
BATCH_SIZE = 20
VALIDATION_SIZE = 500
EPOCH_BOUND = 1000
EARLY_STOP_CHECK_EPOCH = 20
TAKE_CROSS_VALIDATION = False
CROSS_VALIDATION = 5
EPOCH_PER_DECAY = 15
########## Hyperparameter ##########

def weight_variable(shape, w=0.1):
    initial = tf.truncated_normal(shape, stddev=w) #Outputs random values from a truncated normal distribution.
    return tf.Variable(initial)

def bias_variable(shape, w=0.1):
    initial = tf.constant(w, shape=shape)
    return tf.Variable(initial)

def cnn_model(features, labels, mode):
    # Input Layer
    # input layer shape should be [batch_size, image_width, image_height, channels] for conv2d
    # set batch_size = -1 means batch_size = the number of input
    print('input features shape: ', features)
    print('input labels shape: ', labels)
    input_layer = tf.reshape(features, [-1, 32, 32, 3])
    print('input layer shape: ', input_layer.shape)
    # conv1
    with tf.variable_scope('conv1') as scope:
        kernel = weight_variable(shape=[5, 5, 3, 64]) #shape=[filter_height * filter_width * in_channels, output_channels]
        conv = tf.nn.conv2d(input_layer, kernel, strides=[1, 1, 1, 1], padding='SAME')
        biases = bias_variable(shape=[64], w=0.1)
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
    print("conv1 tensor: ", conv1)
    # norm1
    norm1 = tf.nn.lrn(conv1, depth_radius=4, bias=1.0, alpha=0.001/9.0 , beta=0.75, name='norm1')
    print("norm1 tensor: ", norm1) 
    # pool1
    pool1 = tf.nn.max_pool(norm1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
    print("pool1 tensor: ", pool1)

    # conv2
    with tf.variable_scope('conv2') as scope:
        kernel = weight_variable(shape=[5, 5, 64, 64])
        conv = tf.nn.conv2d(pool1, kernel, strides=[1, 1, 1, 1], padding='SAME')
        biases = bias_variable(shape=[64], w=0.1)
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
    print("conv2 tensor: ", conv2)
    # norm2
    norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001/9.0 , beta=0.75, name='norm2')
    print("norm2 tensor: ", norm2)
    # pool2
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')
    print("pool2 tensor: ", pool2) 
    pool2_flat = tf.reshape(pool2, [-1, 8*8*64])
    # conv1
    # with tf.variable_scope('conv1') as scope:
    #     kernel = weight_variable(shape=[5, 5, 3, 32]) #shape=[filter_height * filter_width * in_channels, output_channels]
    #     conv = tf.nn.conv2d(input_layer, kernel, strides=[1, 1, 1, 1], padding='SAME')
    #     biases = bias_variable(shape=[32], w=0.1)
    #     pre_activation = tf.nn.bias_add(conv, biases)
    #     conv1 = tf.nn.elu(pre_activation, name=scope.name)
    # print("conv1 tensor: ", conv1)
    # # # norm1
    # # norm1 = tf.nn.lrn(conv1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
    # # print("norm1 tensor: ", norm1) 
    # # pool1
    # pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')
    # print("pool1 tensor: ", pool1)

    # # conv2
    # with tf.variable_scope('conv2') as scope:
    #     kernel = weight_variable(shape=[5, 5, 32, 48])
    #     conv = tf.nn.conv2d(pool1, kernel, strides=[1, 1, 1, 1], padding='SAME')
    #     biases = bias_variable(shape=[48], w=0.1)
    #     pre_activation = tf.nn.bias_add(conv, biases)
    #     conv2 = tf.nn.elu(pre_activation, name=scope.name)
    # print("conv2 tensor: ", conv2)
    # # # norm2
    # # norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    # # print("norm2 tensor: ", norm2)
    # # pool2
    # pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')
    # print("pool2 tensor: ", pool2) 
    
    # #conv3
    # with tf.variable_scope('conv3') as scope:
    #     kernel = weight_variable(shape=[3, 3, 48, 64])
    #     conv = tf.nn.conv2d(pool2, kernel, strides=[1, 1, 1, 1], padding='SAME')
    #     biases = bias_variable(shape=[64], w=0.1)
    #     pre_activation = tf.nn.bias_add(conv, biases)
    #     conv3 = tf.nn.elu(pre_activation, name=scope.name)
    # print("conv3 tensor: ", conv3)

    # #conv4
    # with tf.variable_scope('conv4') as scope:
    #     kernel = weight_variable(shape=[3, 3, 64, 64])
    #     conv = tf.nn.conv2d(conv3, kernel, strides=[1, 1, 1, 1], padding='SAME')
    #     biases = bias_variable(shape=[64], w=0.1)
    #     pre_activation = tf.nn.bias_add(conv, biases)
    #     conv4 = tf.nn.elu(pre_activation, name=scope.name)
    # print("conv4 tensor: ", conv4)

    # #conv5
    # with tf.variable_scope('conv5') as scope:
    #     kernel = weight_variable(shape=[3, 3, 64, 48])
    #     conv = tf.nn.conv2d(conv4, kernel, strides=[1, 1, 1, 1], padding='SAME')
    #     biases = bias_variable(shape=[48], w=0.1)
    #     pre_activation = tf.nn.bias_add(conv, biases)
    #     conv5 = tf.nn.elu(pre_activation, name=scope.name)
    # print("conv5 tensor: ", conv5)

    # # pool3
    # pool3 = tf.nn.max_pool(conv5, ksize=[1, 2, 2, 1],
    #                      strides=[1, 2, 2, 1], padding='VALID', name='pool3')
    # print("pool3 tensor: ", pool3)

    # # flatten
    # pool3_flat = tf.reshape(pool3, [-1, 4*4*48])

    dense1 = tf.layers.dense(
        inputs=pool2_flat,
        units=1024, # number of neurons in the dense layer
        activation=tf.nn.elu,
        name='dense1')
    dropout1 = tf.layers.dropout(
        inputs=dense1,
        rate=0.1,
        training= mode=='TRAIN',
        name='dropout1')
    dense2 = tf.layers.dense(
        inputs=dropout1,
        units=1024, # number of neurons in the dense layer
        activation=tf.nn.elu,
        name='dense2')
    dropout2 = tf.layers.dropout(
        inputs=dense2,
        rate=0.1,
        training= mode=='TRAIN',
        name='dropout2')

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout2, units=100, name='logits')
    
    return logits

def train(X_train, y_train, X_validate, y_validate, optimizer, epoch_bound, stop_threshold, batch_size, testing=False):

    global saver
    global predictions
    global loss, accuracy
    
    early_stop = 0
    winner_loss = np.infty
    winner_accuracy = 0.0
    
    for epoch in range(epoch_bound):

        # randomize training set
        indices_training = np.random.permutation(X_train.shape[0])
        X_train, y_train = X_train[indices_training,:], y_train[indices_training]

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
        # split validation set into multiple mini-batches and start validating
        cur_loss = 0.0
        cur_accuracy = 0.0
        total_batches = int(X_validate.shape[0] / batch_size)
        for batch in range(total_batches):
            if batch == total_batches - 1:
                cur_loss += sess.run(loss, feed_dict={x:X_validate[batch*batch_size:]
                                                           , y:y_validate[batch*batch_size:]
                                                           , mode:'EVAL'})
                cur_accuracy += sess.run(accuracy, feed_dict={x:X_validate[batch*batch_size:]
                                                           , y:y_validate[batch*batch_size:]
                                                           , mode:'EVAL'})
            else:
                cur_loss += sess.run(loss, feed_dict={x:X_validate[batch*batch_size : (batch+1)*batch_size]
                                                           , y:y_validate[batch*batch_size : (batch+1)*batch_size]
                                                           , mode:'EVAL'})
                cur_accuracy += sess.run(accuracy, feed_dict={x:X_validate[batch*batch_size:]
                                                           , y:y_validate[batch*batch_size:]
                                                           , mode:'EVAL'})
        cur_loss /= total_batches
        cur_accuracy /= total_batches

        # If the accuracy rate does not increase for many times, it will early stop epochs-loop 
        if cur_loss < winner_loss:
            early_stop = 0
            winner_loss = cur_loss
            winner_accuracy = cur_accuracy
            # save best model in testing phase
            if testing == True:
                save_path = saver.save(sess, "./saved_model/cifar-100.ckpt")
        else:
            early_stop += 1
        print('\tEpoch: ', epoch, '\tBest loss: ', winner_loss, '\tLoss: ', cur_loss, '\tAccuracy: ', cur_accuracy)
        if early_stop == stop_threshold:
            break

    return winner_loss, winner_accuracy

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
    y_train, y_validate = np.array(Labels[y_train_idx]), np.array(Labels[y_validate_idx])
    return X_train, y_train, X_validate, y_validate



########## Data ##########
# Data format:
# data -- a 10000x3072 numpy array of uint8s. 
#         Each row of the array stores a 32x32 colour image. 
#         The first 1024 entries contain the red channel values, 
#         the next 1024 the green, and the final 1024 the blue. 
#         The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.
# labels -- a list of 10000 numbers in the range 0-99. The number at index i indicates the label of the ith image in the array data.

(train_data, train_labels), (eval_data, eval_labels) = get_data.load_data(path='./Data/cifar-100/')
train_data = train_data.reshape(train_data.shape[0], 32*32*3)
eval_data = eval_data.reshape(eval_data.shape[0], 32*32*3)

# normalize inputs from 0-255 to 0.0-1.0
train_data = train_data / 255.0
eval_data = eval_data / 255.0

print('Train Data shape: ',train_data.shape)
print('Train Label shape: ', train_labels.shape)

########## Data ##########


########## CNN classifier ##########
x = tf.placeholder(tf.float32, [None, train_data.shape[1]], name='x')
y = tf.placeholder(tf.int32, [None], name='y')
mode = tf.placeholder(tf.string, name='mode')

logits = cnn_model(x, y, mode)

# Calculate Loss (for both TRAIN and EVAL modes)
onehot_labels = tf.one_hot(indices=tf.cast(y, tf.int32), depth=100)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels, logits=logits), name="loss")

# Training iteration (for TRAIN)
## Decaying learning rate exponentially based on the total training step
decay_steps = int(BATCH_SIZE * EPOCH_PER_DECAY) # Define decay steps
global_step = tf.contrib.framework.get_or_create_global_step()

learning_rate = tf.train.exponential_decay(
        learning_rate=0.1, #initial learning rate
        global_step=tf.train.get_global_step(),
        decay_steps=global_step,
        decay_rate=0.96,
        staircase=True,
        name='ExponentialDecayLearningRate'
    )
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate, name='GD_Optimizer')
grads = optimizer.compute_gradients(loss)
apply_gradient_op = optimizer.apply_gradients(grads, global_step=tf.train.get_global_step())
ema = tf.train.ExponentialMovingAverage(decay=0.9999)
maintain_averages_op = ema.apply(tf.trainable_variables())

with tf.control_dependencies([apply_gradient_op, maintain_averages_op]):
      train_op = tf.no_op(name='train_op')

# For prediction (for EVAL)
probabilities = tf.nn.softmax(logits, name="softmax_tensor")
predictions = {
  # Generate predictions (for PREDICT and EVAL mode)
  "classes": tf.argmax(input=probabilities, axis=1),
  # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
  # `logging_hook`.
  "probabilities": probabilities
}

# Calculate Accuracy
correct_prediction = tf.equal(y, tf.argmax(probabilities,1,output_type=tf.int32))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")
########## CNN classifier ##########

########## Train ##########
print("########## Start training ##########")
sess = tf.Session()
init = tf.global_variables_initializer()
# init saver to save model
saver = tf.train.Saver()
writer = tf.summary.FileWriter("logs/", sess.graph)

# randomize dataset
indices = np.random.permutation(train_data.shape[0])

# start cross validation
avg_accuracy = 0.0

if TAKE_CROSS_VALIDATION == True:
    for fold in range(1, CROSS_VALIDATION+1):
        print("########## Fold:", fold, "##########")
        # init weights
        sess.run(init)
        # restore the precious best model
        saver.restore(sess, "./saved_model/cifar-100.ckpt")
        # split inputs into training set and validation set for each fold
        X_train, y_train, X_validate, y_validate = split_folds(indices, train_data, train_labels, CROSS_VALIDATION, fold)
        print('validate data: ', X_validate.shape)
        print('validate label: ', y_validate.shape)
        print('train data: ', X_train.shape)
        print('train label: ', y_train.shape)

        winner_loss = train(X_train, y_train, X_validate, y_validate
                                , train_op, EPOCH_BOUND, EARLY_STOP_CHECK_EPOCH, BATCH_SIZE, testing=False)
        avg_loss += winner_loss
        print("Accuracy:", winner_loss)
    avg_loss /= cross_validation
    print("Average accuracy of cross validation:", avg_accuracy)

# init weights
sess.run(init)
# restore the precious best model
# if os.path.exists("./saved_model/cifar-100.ckpt.meta") == True:
#     print("restore the precious best model")
#     saver.restore(sess, "./saved_model/cifar-100.ckpt")
# else:
#     # init weights
#     sess.run(init)
# randomize dataset
indices = np.random.permutation(train_data.shape[0])
Inputs, Labels = np.array(train_data[indices,:]), np.array(train_labels[indices])

# get validation set with the size of a batch for early-stop
X_train, y_train = Inputs[VALIDATION_SIZE:], Labels[VALIDATION_SIZE:]
X_validate, y_validate = Inputs[:VALIDATION_SIZE], Labels[:VALIDATION_SIZE]
print('X_train: ', X_train.shape, 'y_train: ', y_train.shape)
print('X_validate: ', X_validate.shape, 'y_validate: ', y_validate.shape)


# start training with all the inputs
winner_loss, winner_accuracy = train(X_train, y_train, X_validate, y_validate
                        , train_op, EPOCH_BOUND, EARLY_STOP_CHECK_EPOCH, BATCH_SIZE, testing=True)
print("training with all the inputs, Best loss: ", winner_loss, "Best accuracy: ", winner_accuracy)


sess.close()
########## Train ##########


########## Evaluate ##########
# Evaluate the model and print results
print("########## Start evaluating ##########")
sess = tf.Session()
# restore the precious best model
saver.restore(sess, "./saved_model/cifar-100.ckpt")

testing_accuracy = 0.0
for i in range(5):
    testing_accuracy += sess.run(accuracy, feed_dict={x:eval_data[i*2000:(i+1)*2000], y:eval_labels[i*2000:(i+1)*2000], mode:'EVAL'})

print('Test accuracy: ', testing_accuracy/5)
sess.close()
########## Evaluate ##########


