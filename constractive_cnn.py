# import cnn_functions as cf
import tensorflow as tf


class ConstractiveFourLayers():
    def __init__(self, input_op):
        self.input_op = input_op

    def forward(self):
        # tf.nn.conv2d用法 https://blog.csdn.net/loseinvain/article/details/78935192
        x = self.input_op

        kernel1 = tf.get_variable(shape=[3, 3, 1, 64], dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv1 = tf.nn.conv2d(x, filter=kernel1, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(conv1)
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

        kernel2 = tf.get_variable(shape=[3, 3, 64, 128], dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv2 = tf.nn.conv2d(pool1, filter=kernel2, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(conv2)
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

        kernel3 = tf.get_variable(shape=[3, 3, 128, 256], dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv3 = tf.nn.conv2d(pool2, filter=kernel3, strides=[1, 1, 1, 1], padding='SAME')
        relu3 = tf.nn.relu(conv3)
        pool3 = tf.nn.max_pool(relu3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

        kernel4 = tf.get_variable(shape=[3, 3, 256, 512], dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv4 = tf.nn.conv2d(pool3, filter=kernel4, strides=[1, 1, 1, 1], padding='SAME')
        relu4 = tf.nn.relu(conv4)
        pool4 = tf.nn.max_pool(relu4, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

        # conv1 = cf.conv_op(self.input_op, name="conv1", kh=3, kw=3, n_out=64, dh=1, dw=1)
        # pool1 = cf.mpool_op(conv1, name="pool1", kh=3, kw=3, dh=2, dw=2)
        # conv2 = cf.conv_op(pool1, name="conv2", kh=3, kw=3, n_out=128, dh=1, dw=1)
        # pool2 = cf.mpool_op(conv2, name="pool2", kh=3, kw=3, dh=2, dw=2)
        # conv3 = cf.conv_op(pool2, name="conv3", kh=3, kw=3, n_out=256, dh=1, dw=1)
        # pool3 = cf.mpool_op(conv3, name="pool3", kh=3, kw=3, dh=2, dw=2)
        # conv4 = cf.conv_op(pool3, name="conv4", kh=3, kw=3, n_out=512, dh=1, dw=1)
        # pool4 = cf.mpool_op(conv4, name="pool4", kh=3, kw=3, dh=2, dw=2)

        return pool4

# if __name__ == '__main__':
#     model = ConstractiveFourLayers([112, 112, 3])
#     print(model.forward())