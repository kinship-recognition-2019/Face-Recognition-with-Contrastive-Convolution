import tensorflow as tf


def conv_op(input_op, name, kh, kw, n_out, dh, dw):
    n_in = input_op.shape[-1]
    kernel = tf.get_variable(name="kernel_" + name, shape=[kh, kw, n_in, n_out], dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer_conv2d())
    conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding='VALID')
    return conv


def group_conv_op(input_op, kernel, dh, dw):
    conv_group = []

    for i in range(kernel.shape[0]):
        conv_cur = tf.nn.conv2d(input_op, kernel[i], (1, dh, dw, 1), padding='SAME')
        conv_group.append(conv_cur)

    return conv_group

