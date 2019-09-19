import tensorflow as tf

class Linear():
    def __init__(self, name, in_feature, out_feature):
        self.in_feature = in_feature
        self.out_feature = out_feature
        k = 1. / in_feature
        self.weight = tf.transpose(tf.get_variable(name="weight_" + name, shape=[out_feature, in_feature], dtype=tf.float32,
                                  initializer=tf.random_normal_initializer(mean=0, stddev= k)), perm=[1, 0])
        # self.weight = tf.get_variable(name="weight_"+name, shape=[in_feature, out_feature], dtype=tf.float32,
        #                               initializer=tf.contrib.layers.xavier_initializer_conv2d())
        self.bias = tf.get_variable(name="bias_" + name, shape=[out_feature], dtype=tf.float32,
                                  initializer=tf.random_normal_initializer(mean=0, stddev= k))

    def forward(self, input):
        # print("input", input)

        # print("output", tf.add(tf.tensordot(input, self.weight, [[len(input.shape)-1], [0]]), self.bias))
        # return tf.add(tf.matmul(input, self.weight), self.bias)
        return tf.add(tf.tensordot(input, self.weight, [[len(input.shape)-1], [0]]), self.bias)
        # return tf.add(tf.tensordot(input, self.weight, axes=1), self.bias)
