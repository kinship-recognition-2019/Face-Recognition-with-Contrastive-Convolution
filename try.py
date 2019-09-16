from tensorflow.examples.tutorials.mnist import  input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
import tensorflow as tf

x = tf.placeholder(tf.float32, [None, 784])

w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, w)+b)

y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))
train = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

BATCH_SIZE = 100
TOTAL_STEPS = 5000

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(TOTAL_STEPS+1):
        batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)
        sess.run(train, feed_dict={x: batch_x, y_: batch_y})
        correct_prediction = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_, axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        if step % 100 == 0:
            print(step, sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))