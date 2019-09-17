import tensorflow as tf

class Linear():
    def __init__(self, name, in_feature, out_feature):
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.weight = tf.transpose(tf.get_variable(name="weight_" + name, shape=[out_feature, in_feature], dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d()), perm=[1, 0])
        self.bias = tf.get_variable(name="bias_" + name, shape=[out_feature], dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())

    def forward(self, input):
        return tf.add(tf.tensordot(input, self.weight, [[len(input.shape)-1], [0]]), self.bias)
