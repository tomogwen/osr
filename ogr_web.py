import tensorflow as tf
from create_corpus import *
# import sys
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
import time

# format to send over POST
# first char - 0 = classify, 1 = write training data
# second char - label (as below) - irrelevant if classify
# following chars - data


def classifyName(i):
    return {
        0: "Positive Linear",
        1: "Negative Linear",
        2: "Positive Quadratic",
        3: "Negative Quadratic",
        4: "Sine",
        5: "Cosine"
    }[i]


def labelToOutput(i):
    return {
        0: '100000',
        1: '010000',
        2: '001000',
        3: '000100',
        4: '000010',
        5: '000001'
    }[i]


def writeData(label, data):
    print "Adding training data"
    # write data followed by label to file in training data
    timestr = time.strftime("%Y%m%d-%H%M%S")
    f = open("trainingData/" + timestr, "w")
    for i in range(len(data)):
        f.write(str(data[i]))
    f.write(labelToOutput(label))
    f.close()


# provides functionality to read input from file
def inArray(filename):
    mapFlatTemp = np.ones(784, dtype=np.int)
    f = open(filename, 'r')
    for i in range(784):
        mapFlatTemp[i] = f.read(1)
    return mapFlatTemp


def extractArray(data):
    mapFlatTemp = np.ones(784, dtype=np.int)
    i = 0
    for c in data:
        mapFlatTemp[i] = c
        i += 1
    return mapFlatTemp


def classify(input):
    print "Classify request"
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, 'models/model1')

        mapFlat = extractArray(input)

        feed_dict = {x: [mapFlat], keep_prob: 1.0}
        output = sess.run(y_conv, feed_dict)
        bestGuess = classifyName(np.argmax(output[0]))

        print np.reshape(mapFlat, (28,28))
        print output
        print bestGuess

        return bestGuess


class S(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

    def do_POST(self):
        self._set_headers()

        self.data_string = self.rfile.read(int(self.headers['Content-Length']))
        inputData = self.data_string
        intData = [int(i) for i in inputData]
        print "Recieved: " + inputData
        checkValid = 0
        for i in range(len(intData)):
            if not(0 <= intData[i] and intData[i] <= 9):
                self.wfile.write("datainvalid")
                checkValid = 1
        if intData[0] == 0 and checkValid == 0:
            bestGuess = classify(intData[2:])
            self.wfile.write(bestGuess)

        if intData[0] == 1 and checkValid == 0:
            writeData(intData[1], intData[2:])
            self.wfile.write("datasaved")


def run(server_class=HTTPServer, handler_class=S, port=1234):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print 'Starting httpd...'
    httpd.serve_forever()


mapArray = np.ones((28, 28), dtype=np.int)


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

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

run()


