import tensorflow as tf
import numpy as np


class ConstractiveFourLayers():

    def forward(self, x):

        kernel1 = tf.get_variable(name="kernel1", shape=[3, 3, 1, 64], dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv1 = tf.nn.conv2d(x, filter=kernel1, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(conv1)
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

        kernel2 = tf.get_variable(name="kernel2", shape=[3, 3, 64, 128], dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv2 = tf.nn.conv2d(pool1, filter=kernel2, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(conv2)
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

        kernel3 = tf.get_variable(name="kernel3", shape=[3, 3, 128, 256], dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv3 = tf.nn.conv2d(pool2, filter=kernel3, strides=[1, 1, 1, 1], padding='SAME')
        relu3 = tf.nn.relu(conv3)
        pool3 = tf.nn.max_pool(relu3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

        kernel4 = tf.get_variable(name="kernel4", shape=[3, 3, 256, 512], dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv4 = tf.nn.conv2d(pool3, filter=kernel4, strides=[1, 1, 1, 1], padding='SAME')
        relu4 = tf.nn.relu(conv4)
        pool4 = tf.nn.max_pool(relu4, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

        return pool4


if __name__ == '__main__':
    # [batch, in_height, in_width, in_channels]
    norm = np.random.rand(10, 250, 250, 1)

    sess = tf.Session()


    x = tf.placeholder(tf.float32, [None, 250, 250, 1])
    model = ConstractiveFourLayers()
    ccnn = model.forward(x)

    sess.run(tf.global_variables_initializer())
    print(sess.run(ccnn, feed_dict={x: norm}))
