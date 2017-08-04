import tensorflow as tf
sess = tf.InteractiveSession()

from tensorflow.examples.tutorials.mnist import input_data                      # import training data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)                   # set up data object

x = tf.placeholder(tf.float32, shape=[None, 784])                               # 784 = flattened 28x28 image
y_ = tf.placeholder(tf.float32, shape=[None, 10])                               # 10 = number of possible outputs

W = tf.Variable(tf.zeros([784, 10]))                                            # weights
b = tf.Variable(tf.zeros([10]))                                                 # biases

sess.run(tf.global_variables_initializer())                                     # initalises variables in tf session

y = tf.matmul(x, W) + b                                                         # adds regression model (x*weights+bias)
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y) )              # adds cross entropy loss function

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)     # adds gradient descent

for _ in range(1000):                                                           # training loop
    batch = mnist.train.next_batch(100)                                         # loads 100 training data
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})                       # feed_dict replaces placeholders

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))                # argmax gives index of highest entry

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))              # take mean of converted booleans

print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
