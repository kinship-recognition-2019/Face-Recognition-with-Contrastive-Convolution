import tensorflow as tf


def conv_op(input_op, name, kh, kw, n_out, dh, dw):
    n_in = input_op.shape[-1]
    kernel = tf.get_variable(name="kernel_" + name, shape=[kh, kw, n_in, n_out], dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer_conv2d())
    conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding='VALID')
    return conv


def group_conv_op(input_op, name, kh, kw, n_out, dh, dw, groups):
    pass
    # n_out_final = n_out / groups
    # inputs = tf.split(input_op, groups, -1)
    # for i in range(len(inputs)):
    #     conv_cur = conv_op(inputs[i], name, kh, kw, n_out_final, dh, dw)




# if __name__ == '__main__':
#     norm = np.random.rand(10, 128, 128, 1)
#
#     sess = tf.Session()
#
#     x = tf.placeholder(tf.float32, [None, 128, 128, 1])
#     model =
#     ccnn = model.forward(x, "zz")
#
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(ccnn, feed_dict={x: norm}))
#     print(ccnn)