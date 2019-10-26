import tensorflow as tf


def conv_op(input_op, name, kh, kw, n_out, dh, dw):
    n_in = input_op.shape[-1]
    kernel = tf.get_variable(name="kernel_" + name, shape=[kh, kw, n_in, n_out], dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer_conv2d())
    conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding='VALID')
    print("kernel:")
    print(kernel)
    print("input_op:")
    print(input_op)
    return conv

def convolution(k, data):
    n,m,channels = data.shape
    img_new = []
    for l in range(channels):
        img_new_channel=[]
        for i in range(n-3):
            line = []
            for j in range(m-3):
                a = data[:,:,l][i:i+3,j:j+3]
                line.append(np.sum(np.multiply(k[:,:,l], a)))
            img_new_channel.append(line)
        img_new.append(img_new_channel)
    return np.array(img_new)
def weights_initializer(kernel):
    return kernel;
"""
def convolution(k,data):
    n,m,channels=data.shape
    img_new=[]
    for l in range(channels):
        img_new_channel=[]
        for i in range(n-3):
            line=[]
"""            
def group_conv_op(input_op, kernel, dh, dw, groups):
    # n_in = input_op.shape[-1] / groups
    # n_out = n_out / groups
    # print("input_op", input_op)
    inputs = tf.split(input_op, groups, 3)
    kernels = tf.split(kernel, groups, 2)
    kernels = tf.stop_gradient(kernels)
    # print("inputs", inputs)
    # print("kernel", kernel)
    # print("sz", len(inputs))
    conv_group = []
    #print(inputs)
    for i in range(len(inputs)):
        conv_cur = tf.nn.conv2d(inputs[i], kernels[i], (1, dh, dw, 1), padding='SAME')
        #bs,w,h,c=kernels[i].shape
        #conv_cur = tf.contrib.layers.conv2d(inputs=inputs[i],num_outputs=c,kernel_size=(1,dh,dw,1),padding="SAME",weights_initializer=weights_initializer(kernels[i]),trainable=False)
        conv_group.append(conv_cur)
    return conv_group, inputs



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
