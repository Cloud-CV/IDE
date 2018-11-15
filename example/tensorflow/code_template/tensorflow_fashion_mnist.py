from __future__ import print_function

import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist

(train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()

train_shape = train_x[0].shape

tf.reset_default_graph()


# convolution net initializator using low-level API of tensorflow
def conv_net(x, keep_prob):
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


def train_neural_network(session, optimizer, keep_probability, feature_batch, label_batch):
    session.run(optimizer,
                feed_dict={
                    x: feature_batch,
                    y: label_batch,
                    keep_prob: keep_probability
                })


def print_stats(sess, feature_batch, label_batch,
                valid_features, valid_labels, cost, accuracy):
    loss = sess.run(cost,
                    feed_dict={
                        x: feature_batch,
                        y: label_batch,
                        keep_prob: 1.
                    })
    valid_acc = sess.run(accuracy,
                         feed_dict={
                             x: valid_features,
                             y: valid_labels,
                             keep_prob: 1.
                         })

    print('Loss: {:>10.4f} Validation Accuracy: {:.6f}'.format(loss, valid_acc))


epochs = 10
batch_size = 128
n_batches = train_x.shape[0] // batch_size + 1
keep_probability = 0.7
learning_rate = 0.001

x = tf.placeholder(tf.float32, shape=(None, 28, 28), name='input_x')
depth = 10
y = tf.placeholder(tf.uint8, shape=(None,), name='output_y')
check_y = tf.one_hot(y, depth)
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

logits = conv_net(x, keep_prob)

# cost is loss function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=check_y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(check_y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Training cycle
    for epoch in range(epochs):
        for batch_i in range(n_batches):
            batch_start = batch_i*batch_size
            batch_end = batch_start+batch_size
            batch_features = train_x[batch_start:batch_end]
            batch_labels = train_y[batch_start:batch_end]

            train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)

            print('Epoch {:>2}, Fashion MNIST Batch {}:  '.format(epoch + 1, batch_i + 1), end='')
            print_stats(sess, batch_features, batch_labels, test_x, test_y, cost, accuracy)
