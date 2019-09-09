import tensorflow as tf


def extract_patches(x, patch_size):
    # unfold(dim, size, step) → Tensor
    patches = x.unfold(2, patch_size, 1).unfold(3, patch_size, 1)
    bs, c, pi, pj, _, _ = patches.size()

    l = [patches[:, :, int(i / pi), i % pi, :, :] for i in range(pi * pi)]
    f = [l[i].contiguous().view(-1, c * patch_size * patch_size) for i in range(pi * pi)]

    stack_tensor = tf.stack(f)

    stack_tensor = stack_tensor.permute(1, 0, 2)
    return stack_tensor


class GenModel():
    def __init__(self, feature_size):
        self.f_size = feature_size

    def linear(self, input_size, output_size):
        a, b, c = x.shape()
        w = tf.get_variable(name="w", shape=[b, a], dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())

    def forward(self, x):
        S0 = x

        kernel1 = tf.get_variable(name="kernel1", shape=[3, 3, self.f_size, self.f_size], dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv1 = tf.nn.conv2d(S0, filter=kernel1, strides=[1, 1, 1, 1], padding='SAME')
        S1 = tf.nn.relu(conv1)

        kernel2 = tf.get_variable(name="kernel2", shape=[3, 3, self.f_size, self.f_size], dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv2 = tf.nn.conv2d(S1, filter=kernel2, strides=[1, 1, 1, 1], padding='SAME')
        S2 = tf.nn.relu(conv2)

        p1 = extract_patches(S0, 3)
        p2 = extract_patches(S1, 2)
        p3 = extract_patches(S2, 1)






