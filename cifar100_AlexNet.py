
# coding: utf-8

# In[ ]:


import tensorflow as tf
import numpy as np
import os
import get_data
import data_augmentation


tf.logging.set_verbosity(tf.logging.INFO)

########## Hyperparameter ##########
BATCH_SIZE = 20
VALIDATION_SIZE = 500
EPOCH_BOUND = 1000
EARLY_STOP_CHECK_EPOCH = 20
TAKE_CROSS_VALIDATION = False
CROSS_VALIDATION = 5
EPOCH_PER_DECAY = 10
########## Hyperparameter ##########

def weight_variable(shape, w=0.1):
    initial = tf.truncated_normal(shape, stddev=w) #Outputs random values from a truncated normal distribution.
    return tf.Variable(initial)

def bias_variable(shape, w=0.1):
    initial = tf.constant(w, shape=shape)
    return tf.Variable(initial)

def mean_var_with_update(ema, fc_mean, fc_var):
    ema_apply_op = ema.apply([fc_mean, fc_var])
    with tf.control_dependencies([ema_apply_op]):
        return tf.identity(fc_mean), tf.identity(fc_var)

def Conv2d(scope_name, inputs, filter_size, in_channels, output_channels, strides_moves, padding_mode, activation):
    with tf.variable_scope(scope_name) as scope:
        kernel = weight_variable(shape=[filter_size, filter_size, in_channels, output_channels])
        conv = tf.nn.conv2d(inputs, kernel, strides=[1, strides_moves, strides_moves, 1], padding=padding_mode)
        biases = bias_variable(shape=[output_channels], w=0.1)
        pre_activation = tf.nn.bias_add(conv, biases)
    return activation(pre_activation, name=scope.name)

def Conv2d_BN(scope_name, inputs, filter_size, in_channels, output_channels, strides_moves, padding_mode, activation):
    with tf.variable_scope(scope_name) as scope:
        kernel = weight_variable(shape=[filter_size, filter_size, in_channels, output_channels])
        conv = tf.nn.conv2d(inputs, kernel, strides=[1, strides_moves, strides_moves, 1], padding=padding_mode)
        biases = bias_variable(shape=[output_channels], w=0.1)
        pre_activation = tf.nn.bias_add(conv, biases)
        fc_mean, fc_var = tf.nn.moments(pre_activation, axes=[0, 1, 2])
        scale = tf.Variable(tf.ones([output_channels]))
        shift = tf.Variable(tf.zeros([output_channels]))
        epsilon = 0.001
        ema = tf.train.ExponentialMovingAverage(decay=0.5)
        # update mean and var when the value of mode is TRAIN, or back to the previous Moving Average of fc_mean and fc_var 
        mean, var = tf.cond(train_mode, lambda: (mean_var_with_update(ema, fc_mean, fc_var)), lambda: (ema.average(fc_mean), ema.average(fc_var)))
        pre_activation_with_BN = tf.nn.batch_normalization(pre_activation, mean, var, shift, scale, epsilon)
    return activation(pre_activation_with_BN, name=scope.name)

def cnn_model(features, labels, train_mode):
    # Input Layer
    # input layer shape should be [batch_size, image_width, image_height, channels] for conv2d
    # set batch_size = -1 means batch_size = the number of input
    print('input features shape: ', features)
    print('input labels shape: ', labels)
    input_layer = tf.reshape(features, [-1, 32, 32, 3])
    print('input layer shape: ', input_layer.shape)

    # AlexNet
    # conv1
    conv1 = Conv2d_BN('conv1', input_layer, 11, 3, 96, 4, 'SAME', tf.nn.relu)
    print("conv1 tensor: ", conv1)
    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')
    print("pool1 tensor: ", pool1)
    # conv2
    conv2 = Conv2d_BN('conv2', pool1, 5, 96, 256, 1, 'SAME', tf.nn.relu)
    print("conv2 tensor: ", conv2)
    # pool2
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')
    print("pool2 tensor: ", pool2)
    # conv3
    conv3 = Conv2d_BN('conv3', pool2, 3, 256, 384, 1, 'SAME', tf.nn.relu)
    print("conv3 tensor: ", conv3)
    # conv4
    conv4 = Conv2d_BN('conv4', conv3, 3, 384, 384, 1, 'SAME', tf.nn.relu)
    print("conv4 tensor: ", conv4)
    # conv5
    conv5 = Conv2d_BN('conv5', conv4, 3, 384, 256, 1, 'SAME', tf.nn.relu)
    print("conv5 tensor: ", conv5)
    # pool3
    pool3 = tf.nn.max_pool(conv5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool3')
    print("pool3 tensor: ", pool3)
    # flatten
    pool3_flat = tf.reshape(pool3, [-1, 1*1*256])

    dense1 = tf.layers.dense(
        inputs=pool3_flat,
        units=128, # number of neurons in the dense layer
        activation=tf.nn.relu,
        name='dense1')
    dropout1 = tf.layers.dropout(
        inputs=dense1,
        rate=0.5,
        training= train_mode,
        name='dropout1')
    dense2 = tf.layers.dense(
        inputs=dropout1,
        units=128, # number of neurons in the dense layer
        activation=tf.nn.relu,
        name='dense2')
    dropout2 = tf.layers.dropout(
        inputs=dense2,
        rate=0.5,
        training= train_mode,
        name='dropout2')

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout2, units=100, name='logits')
    
    return logits

def train(X_train, y_train, X_validate, y_validate, train_op, epoch_bound, stop_threshold, batch_size, testing=False):

    global saver
    global predictions
    global loss, accuracy_top1
    global writer, merged
    
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
                sess.run(train_op, feed_dict={x: X_train[batch*batch_size:], 
                                               y: y_train[batch*batch_size:], 
                                               train_mode: True})
                summary = sess.run(merged, feed_dict={x: X_train[batch*batch_size:],
                                                      y: y_train[batch*batch_size:],
                                                      train_mode: True})
                writer.add_summary(summary, epoch + (batch/total_batches))
            else:
                sess.run(train_op, feed_dict={x: X_train[batch*batch_size : (batch+1)*batch_size], 
                                               y: y_train[batch*batch_size : (batch+1)*batch_size], 
                                               train_mode: True})
                summary = sess.run(merged, feed_dict={x: X_train[batch*batch_size : (batch+1)*batch_size],
                                                      y: y_train[batch*batch_size : (batch+1)*batch_size],
                                                      train_mode: True})
                writer.add_summary(summary, epoch + (batch/total_batches))
        # split validation set into multiple mini-batches and start validating
        cur_loss = 0.0
        cur_accuracy = 0.0
        total_batches = int(X_validate.shape[0] / batch_size)
        for batch in range(total_batches):
            if batch == total_batches - 1:
                cur_loss += sess.run(loss, feed_dict={x:X_validate[batch*batch_size:]
                                                           , y:y_validate[batch*batch_size:]
                                                           , train_mode: False})
                cur_accuracy += sess.run(accuracy_top1, feed_dict={x:X_validate[batch*batch_size:]
                                                           , y:y_validate[batch*batch_size:]
                                                           , train_mode: False})
            else:
                cur_loss += sess.run(loss, feed_dict={x:X_validate[batch*batch_size : (batch+1)*batch_size]
                                                           , y:y_validate[batch*batch_size : (batch+1)*batch_size]
                                                           , train_mode: False})
                cur_accuracy += sess.run(accuracy_top1, feed_dict={x:X_validate[batch*batch_size:]
                                                           , y:y_validate[batch*batch_size:]
                                                           , train_mode: False})
        cur_loss /= total_batches
        cur_accuracy /= total_batches

        # If the accuracy rate does not increase for many times, it will early stop epochs-loop 
        if cur_loss < winner_loss:
            early_stop = 0
            winner_loss = cur_loss
            winner_accuracy = cur_accuracy
            # save best model in testing phase
            if testing == True:
                save_path = saver.save(sess, "./saved_model/cifar100_AlexNet/cifar-100_AlexNet.ckpt")
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

# load cifar data
# (train_data, train_labels), (eval_data, eval_labels) = get_data.load_data(path='./Data/cifar-100/')
# train_data = train_data.reshape(train_data.shape[0], 32*32*3)
# eval_data = eval_data.reshape(eval_data.shape[0], 32*32*3)

# load cifar data and data augmentation
(train_data, train_labels), (eval_data, eval_labels) = data_augmentation.generate()
train_data = train_data.reshape(train_data.shape[0], 32*32*3)
eval_data = eval_data.reshape(eval_data.shape[0], 32*32*3)

# data augmentation


# normalize inputs from 0-255 to 0.0-1.0
train_data = train_data / 255.0
eval_data = eval_data / 255.0

print('Train Data shape: ',train_data.shape)
print('Train Label shape: ', train_labels.shape)

########## Data ##########


########## CNN classifier ##########
x = tf.placeholder(tf.float32, [None, train_data.shape[1]], name='x')
y = tf.placeholder(tf.int32, [None], name='y')
train_mode = tf.placeholder(tf.bool, name='train_mode')

logits = cnn_model(x, y, train_mode)

# Calculate Loss (for both TRAIN and EVAL modes)
onehot_labels = tf.one_hot(indices=tf.cast(y, tf.int32), depth=100)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels, logits=logits), name="loss")
tf.summary.scalar('loss', loss)

# Training iteration (for TRAIN)
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
train_op = optimizer.minimize(loss, name='train_op')
## exponential moving average
ema = tf.train.ExponentialMovingAverage(decay=0.9999)

with tf.control_dependencies([train_op]):
      train_op = ema.apply(tf.trainable_variables()) # apply exponential moving average to training operation

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
accuracy_top1 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions=probabilities, targets=y, k=1), tf.float32), name="accuracy_top1")
accuracy_top3 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions=probabilities, targets=y, k=3), tf.float32), name="accuracy_top3")
accuracy_top5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions=probabilities, targets=y, k=5), tf.float32), name="accuracy_top5")
########## CNN classifier ##########

########## Train ##########
print("########## Start training ##########")
sess = tf.Session()
init = tf.global_variables_initializer()
# init saver to save model
saver = tf.train.Saver()

# visualize data
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("logs/cifar100_AlexNet", sess.graph)

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
        saver.restore(sess, "./saved_model/cifar100_AlexNet/cifar-100_AlexNet.ckpt")
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
saver.restore(sess, "./saved_model/cifar100_AlexNet/cifar-100_AlexNet.ckpt")

testing_accuracy_top1 = 0.0
testing_accuracy_top3 = 0.0
testing_accuracy_top5 = 0.0
for i in range(5):
    testing_accuracy_top1 += sess.run(accuracy_top1, feed_dict={x:eval_data[i*2000:(i+1)*2000], y:eval_labels[i*2000:(i+1)*2000], train_mode: False})
    testing_accuracy_top3 += sess.run(accuracy_top3, feed_dict={x:eval_data[i*2000:(i+1)*2000], y:eval_labels[i*2000:(i+1)*2000], train_mode: False})
    testing_accuracy_top5 += sess.run(accuracy_top5, feed_dict={x:eval_data[i*2000:(i+1)*2000], y:eval_labels[i*2000:(i+1)*2000], train_mode: False})

print('Test Top-1 Accuracy: ', testing_accuracy_top1/5, 'Test Top-3 Accuracy: ', testing_accuracy_top3/5, 'Test Top-5 Accuracy: ', testing_accuracy_top5/5)
sess.close()
########## Evaluate ##########


