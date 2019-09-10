import tensorflow as tf
from constractive_cnn import ConstractiveFourLayers
import numpy as np


def extract_patches(x, patch_size):
    cur = tf.extract_image_patches(images=x, ksizes=[1, patch_size, patch_size, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding="VALID")
    dim, channel = int(cur.shape[1]) * int(cur.shape[2]), int(x.shape[3])
    cur = tf.reshape(cur, (-1, dim, channel*patch_size*patch_size))
    # print(cur.shape)
    return cur


class Linear():
    def __init__(self, name, in_feature, out_feature):
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.weight = tf.transpose(tf.get_variable(name="weight_" + name, shape=[out_feature, in_feature], dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d()), perm=[1, 0])
        self.bias = tf.get_variable(name="bias_" + name, shape=[out_feature], dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())

    def forward(self, input):
        return tf.add(tf.tensordot(input, self.weight, [[2], [0]]), self.bias)


class GenModel():
    def __init__(self, feature_size):
        self.f_size = feature_size
        self.g1 = Linear("g1", self.f_size*3*3, self.f_size*3*3)
        self.g2 = Linear("g2", self.f_size*2*2, self.f_size*3*3)
        self.g3 = Linear("g3", self.f_size*1*1, self.f_size*3*3)

    def forward(self, x):
        S0 = x
        kernel1_ = tf.get_variable(name="kernel1_", shape=[3, 3, self.f_size, self.f_size], dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv1 = tf.nn.conv2d(S0, filter=kernel1_, strides=[1, 1, 1, 1], padding='SAME')
        S1 = tf.nn.relu(conv1)

        kernel2_ = tf.get_variable(name="kernel2_", shape=[3, 3, self.f_size, self.f_size], dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv2 = tf.nn.conv2d(S1, filter=kernel2_, strides=[1, 1, 1, 1], padding='SAME')
        S2 = tf.nn.relu(conv2)
        print(S0)
        print(S1)
        print(S2)

        p1 = extract_patches(S0, 3)
        p2 = extract_patches(S1, 2)
        p3 = extract_patches(S2, 1)

        kk1 = tf.nn.relu(self.g1.forward(p1))
        kk2 = tf.nn.relu(self.g2.forward(p2))
        kk3 = tf.nn.relu(self.g3.forward(p3))

        kernels = tf.concat((kk1, kk2, kk3), 1, "kernels")
        return kernels


class Regressor():
    def __init__(self, n):
        self.n = n
        self.linear = Linear("linear", n, 1)

    def forward(self, x):
        bs, c = x.size()
        x = self.linear.forward(x)
        x = tf.sigmoid(x)
        return x


class IdentityRegressor():
    def __init__(self, n, classes):
        self.fc1 = Linear("fc1", n, 256)
        self.fc2 = Linear("fc2", 256, classes)

    def forward(self, x):
        bs, m, n = x.size()
        x = tf.reshape(-1, n*m)
        x = tf.nn.relu(self.fc1.forward(x))
        x = self.fc2.forward(x)
        return x


def main():
    # num_classes = 10574
    # base_model = constractive_cnn.ConstractiveFourLayers()
    # gen_model = GenModel(512)
    # reg_model = Regressor(686)
    # idreg_model = IdentityRegressor(14 * 512 * 3 * 3, num_classes)
    pass


if __name__ == '__main__':

    rd = np.random.rand(10, 250, 250, 1)

    sess = tf.Session()

    x = tf.placeholder("float", [None, 250, 250, 1])

    base_model = ConstractiveFourLayers()
    genarator_model = GenModel(512)

    process = genarator_model.forward(base_model.forward(x))

    sess.run(tf.global_variables_initializer())
    print(sess.run(process, feed_dict={x: rd}))


