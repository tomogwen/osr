import tensorflow as tf
from create_corpus import *

temp = generateCorpus()
corpus = temp[0]
corpusSize = temp[1]


def weight_variable(shape):                                                     # weight var func
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):                                                       # bias var func
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):                                                               # define 2d convolution
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):                                                            # define pooling
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides = [1, 2, 2, 1],
                          padding='SAME')


x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 6])

# first convolutional layer

W_conv1 = weight_variable([5, 5, 1, 32])                                        # 32 outputs per 5x5 patch
b_conv1 = bias_variable([32])                                                   # bias, one per output

x_image = tf.reshape(x, [-1, 28, 28, 1])                                        # reshape to 4 dimensions
                                                                                # [-1, width, height, colours]

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)                        # convolve w/ weight+bias, apply RELU
h_pool1 = max_pool_2x2(h_conv1)                                                 # first pooling, image now 14x14

# second convolutional layer

W_conv2 = weight_variable([5, 5, 32, 64])                                       # 32 inputs, 64 outputs
b_conv2 = bias_variable([64])                                                   # bias, one per output

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)                        # second convolution
h_pool2 = max_pool_2x2(h_conv2)                                                 # second pooling, image now 7x7

# densely connected layer (fully connected)

W_fc1 = weight_variable([7*7*64, 1024])                                         # fc layer weights
b_fc1 = bias_variable([1024])                                                   # fc layer bias

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])                                # flatten layer
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)                      # apply weights/bias, apply relu neuron

# dropout (prevents overfitting by dropping neurons -> taking average)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)                                    # nn.dropout handles it all

# post-dropout layer (readout layer)

W_fc2 = weight_variable([1024, 6])
b_fc2 = bias_variable([6])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2                                   # regression layer

# train and evaluate

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))          # cross_entropy set up
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()
saveState = 0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        batch = nextBatch(corpus, 50)
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
            if train_accuracy > 0.95 and saveState == 0:
                saver.save(sess, 'models/model1')
                saveState = 1
            elif train_accuracy > 0.99 and saveState == 1:
                saver.save(sess, 'models/model1')
                saveState = 2
            elif train_accuracy > 0.995 and saveState == 2:
                saver.save(sess, 'models/model1')
                exit()
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob:0.5})

    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: corpus[0], y_: corpus[1], keep_prob: 1.0}))
    saver.save(sess, 'model1')
