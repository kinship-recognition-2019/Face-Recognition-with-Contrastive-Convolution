import tensorflow as tf
from linear import Linear

def extract_patches(x, patch_size):
    cur = tf.extract_image_patches(images=x, ksizes=[1, patch_size, patch_size, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding="VALID")

    dim, channel = int(cur.shape[1]) * int(cur.shape[2]), int(x.shape[3])
    cur = tf.reshape(cur, (-1, dim, channel*patch_size*patch_size))
    return cur


class GenModel():
    def __init__(self, feature_size):
        self.f_size = feature_size

        self.g1 = Linear("g1", self.f_size*3*3, self.f_size*3*3)
        self.g2 = Linear("g2", self.f_size*2*2, self.f_size*3*3)
        self.g3 = Linear("g3", self.f_size*1*1, self.f_size*3*3)

    def forward(self, x, scope):
        S0 = x

        kernel1_ = tf.get_variable(name="kernel1_"+scope, shape=[3, 3, self.f_size, self.f_size], dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv1 = tf.nn.conv2d(S0, filter=kernel1_, strides=[1, 1, 1, 1], padding='VALID')
        S1 = tf.nn.relu(conv1)

        kernel2_ = tf.get_variable(name="kernel2_"+scope, shape=[3, 3, self.f_size, self.f_size], dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv2 = tf.nn.conv2d(S1, filter=kernel2_, strides=[1, 1, 1, 1], padding='VALID')
        S2 = tf.nn.relu(conv2)

        p1 = extract_patches(S0, 3)
        # print("p1", p1) # bs*9*4608

        p2 = extract_patches(S1, 2)
        # print("p2", p2) # bs*4*2048

        p3 = extract_patches(S2, 1)
        # print("p3", p3) # bs*1*512

        # kernel4_ = tf.get_variable(name="kernel4_"+scope, shape=[1, 1, self.f_size, self.f_size], dtype=tf.float32,
        #                           initializer=tf.contrib.layers.xavier_initializer_conv2d())
        kk1 = tf.nn.relu(self.g1.forward(p1))
        # print("kk1", kk1) # bs*9*4608

        kk2 = tf.nn.relu(self.g2.forward(p2))
        # print("kk2", kk2) # bs*4*4608

        kk3 = tf.nn.relu(self.g3.forward(p3))
        # print("kk3", kk3) # bs*1*4608

        kernels = tf.concat((kk1, kk2, kk3), 1, "kernels")

        return kernels