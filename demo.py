#digit classifier
from __future__ import print_function

import tensorflow as tf
 

#dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data/' one_hot = true)


#use of one_shot
# [house, car, tooth, car]
# It can be represented as [0, 1, 2, 1]
# car < tooth its wrong
# so right way to  represent this
# house=[1,0,0,0]
# car=[0,1,0,0]

learning_rate = 0.001
training_iter = 200000
batch_size = 128 #128 samples
display_step = 10


#network parameters
#28*28 pixel image
n_input = 784
n_classes = 10
dropout = 0.75 #prevents overfitting

#represent gateway
x = tf.placeholder(tf.float32, [ None, n_input]) #for image
y = tf.placeholder(tf.float32, [ None, n_classes]) #for label
keep_prob = tf.placeholder(tf.float32) #for dropout


#define convulational layer
def conv2d(x, W, b, strides):
	x= tf.nn.conv2d(x, W, strides=[1, strides, 1]),padding='SAME')#strides list of integer =tensor = data
	x= tf.nn.bias_add(x, b)
	return tf.nn.relu(x) #relu activation function

def maxpool2d(x, k=2):
	return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides = [1, k, k, 1], padding = 'SAME')


#create model

def conv_net(x, weights, biases, dropout):
	#reshape input data

	x = tf.reshape(x, shape = [-1, 28, 28, 1] )

	#convulational layer
	conv1 = conv2d(x, weights['wc1'], biases['bc1'])
	#max pooling
	conv1 = maxpool2d(conv1, weights['wc2'], biases['bc2'])

	conv2 = conv2dd(conv1, weights['wc2'], biases['bc2'])
	conv2 = maxpool2d(conv2,k=2)

	#fully connected layer 
	fc1 = tf.reshape(conv2, [-1, weights['wd1'].getshape().aslist()])
	fc1 = tf.add(tf.matmul(fc1, weights['wd1'], biases['bc2']))
	fc1 = tf.nn.relu(fc1)

	#apply dropout
	fc1 = tf.nn.dropout(fc1, dropout)

	#output, class prediction
	out = tf.add(tf.matmul(fc1, weights['out'], biases['out']]))
	return out


#create weights
weights = {
	
	'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),# 5*5 1 input 32 output(bits)
	'wc1': tf.Variable(tf.random_normal([5, 5, 32, 64])), #5*5 input 32 output 64
	'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])), #input and output
	'out': tf.Variable(tf.random_normal([1024, n_classes])) #input and output

}	

#construct model

pred = conv_net(x, weights, biasesm keep_prob)

#define optimizer and loss
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logistic())
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

#evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y,1))
accuract = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

#initialize the variables
init = tf.initialize_all_variables()

#launch the graph

with tf.Session as see:
	sess.run(init)
	step = 1

	#keep training until max iteration
	while step * batch_size < training_iter:
		sess.run(optimizer, feed_dict{x: batch_x: y :batch_y})

		print('iteration step')







