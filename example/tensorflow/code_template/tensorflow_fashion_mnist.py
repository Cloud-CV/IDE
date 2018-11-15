from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tqdm import tqdm  # progress bars
import numpy as np


# convolution net initializator using low-level API of tensorflow
def conv_net(x, keep_prob, train_shape):
    shape_x = tf.reshape(x, [-1] + list(train_shape) + [1])

    conv1_filter = tf.Variable(tf.truncated_normal(shape=[1, 1, 1, 64], mean=0, stddev=0.08))
    conv2_filter = tf.Variable(tf.truncated_normal(shape=[1, 1, 64, 128], mean=0, stddev=0.08))

    conv1 = tf.nn.conv2d(shape_x, conv1_filter, strides=[1, 1, 1, 1], padding='SAME')
    conv1 = tf.nn.relu(conv1)
    conv1_pool = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1_bn = tf.layers.batch_normalization(conv1_pool)

    conv2 = tf.nn.conv2d(conv1_bn, conv2_filter, strides=[1, 1, 1, 1], padding='SAME')
    conv2 = tf.nn.relu(conv2)
    conv2_pool = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2_bn = tf.layers.batch_normalization(conv2_pool)

    flat = tf.contrib.layers.flatten(conv2_bn)

    dense = tf.layers.dense(flat, 128, activation=tf.nn.tanh)
    dense = tf.nn.dropout(dense, keep_prob)
    dense = tf.layers.batch_normalization(dense)

    out = tf.layers.dense(dense, 10, activation=None)
    return out


# Train neural network over batch
def train_neural_network(sess, optimizer, keep_probability, feature_batch, label_batch):
    sess.run(optimizer,
             feed_dict={
                 x: feature_batch,
                 y: label_batch,
                 keep_prob: keep_probability
             })


# get loss over batch
def get_batch_loss(sess, feature_batch, label_batch, cost):
    return sess.run(cost,
                    feed_dict={
                        x: feature_batch,
                        y: label_batch,
                        keep_prob: 1.
                    })


# get accuracy over batch
def get_batch_acc(sess, feature_batch, label_batch, accuracy):
    return sess.run(accuracy,
                    feed_dict={
                        x: feature_batch,
                        y: label_batch,
                        keep_prob: 1.
                    })


# writes stats of loss and accuracy over certain dataset
def metrics_summary(dataset, losses, acc):
    return '{0} loss: {1:.4f}, {0} acc: {2:.4f}'.format(dataset, losses.mean(), acc.mean())


(train_x, train_y), (valid_x, valid_y) = fashion_mnist.load_data()

train_shape = train_x[0].shape

epochs = 10
batch_size = 128
keep_probability = 0.7
learning_rate = 0.001

tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape=(None, 28, 28), name='input_x')
depth = 10
y = tf.placeholder(tf.uint8, shape=(None,), name='output_y')
one_hot_y = tf.one_hot(y, depth)
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

logits = conv_net(x, keep_prob, train_shape)

# cost is loss function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

train_n_batches = train_x.shape[0] // batch_size + 1
valid_n_batches = valid_x.shape[0] // batch_size + 1
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Training cycle
    for epoch in range(epochs):
        train_losses = np.zeros(train_n_batches)
        train_accs = np.zeros(train_n_batches)
        valid_losses = np.zeros(valid_n_batches)
        valid_accs = np.zeros(valid_n_batches)

        print('Epoch %d' % (epoch + 1))
        for batch_i in tqdm(range(train_n_batches), desc="Train", ncols=80):
            batch_start = batch_i*batch_size
            batch_end = batch_start+batch_size
            train_features = train_x[batch_start:batch_end]
            train_labels = train_y[batch_start:batch_end]

            train_neural_network(sess, optimizer, keep_probability, train_features, train_labels)

            train_losses[batch_i] = get_batch_loss(sess, train_features, train_labels, cost)
            train_accs[batch_i] = get_batch_acc(sess, train_features, train_labels, accuracy)

        for batch_i in tqdm(range(valid_n_batches), desc="Valid", ncols=80):
            batch_start = batch_i*batch_size
            batch_end = batch_start+batch_size
            valid_features = valid_x[batch_start:batch_end]
            valid_labels = valid_y[batch_start:batch_end]

            valid_losses[batch_i] = get_batch_loss(sess, valid_features, valid_labels, cost)
            valid_accs[batch_i] = get_batch_acc(sess, valid_features, valid_labels, accuracy)

        print('Epoch %d summary:' % (epoch + 1),
              metrics_summary('train', train_losses, train_accs) + ',',
              metrics_summary('valid', valid_losses, valid_accs))
