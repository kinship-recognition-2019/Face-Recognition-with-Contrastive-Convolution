import tensorflow as tf
import numpy as np
from conv_functions import conv_op


class ConstractiveFourLayers():

    def forward(self, x, scope):
        conv1 = conv_op(input_op=x, name="conv1_"+scope, kh=3, kw=3, n_out=64, dh=1, dw=1)
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
        relu1 = tf.nn.relu(pool1)

        conv2 = conv_op(input_op=relu1, name="conv2_"+scope, kh=3, kw=3, n_out=128, dh=1, dw=1)
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
        relu2 = tf.nn.relu(pool2)

        conv3 = conv_op(input_op=relu2, name="conv3_"+scope, kh=3, kw=3, n_out=256, dh=1, dw=1)
        pool3 = tf.nn.max_pool(conv3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
        relu3 = tf.nn.relu(pool3)

        conv4 = conv_op(input_op=relu3, name="conv4_"+scope, kh=3, kw=3, n_out=512, dh=1, dw=1)
        pool4 = tf.nn.max_pool(conv4, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
        relu4 = tf.nn.relu(pool4)

        return relu4


if __name__ == '__main__':
    # [batch, in_height, in_width, in_channels]
    norm = np.random.rand(10, 128, 128, 1)

    sess = tf.Session()

    x = tf.placeholder(tf.float32, [None, 128, 128, 1])
    model = ConstractiveFourLayers()
    ccnn = model.forward(x, "zz")

    sess.run(tf.global_variables_initializer())
    print(sess.run(ccnn, feed_dict={x: norm}))
    print(ccnn)