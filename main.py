import tensorflow as tf
import constractive_cnn

def extract_patches(x, patch_size):
    # tf.slice(input, begin, size, name)
    len = x.shape()[2]
    for i in len - 2:
        for j in len - 2:
            cur = tf.slice(x, [0, 0, ])




    # unfold(dim, size, step) → Tensor
    # slice(input_, begin, size, name=None)
    # patches = x.unfold(2, patch_size, 1).unfold(3, patch_size, 1)
    # start = []
    # for a, b in itertools.product(range(x.shape[2]-patch_size), range(x.shape[3]-patch_size)):
    #     start.append([1, 1, a, b])
    # print(start)
    # print(x.shape)
    # patches = tf.strided_slice(x, [0, 0, 0, 0], [1, 1, patch_size, patch_size])

    # bs, c, pi, pj, _, _ = patches.size()
    #
    # l = [patches[:, :, int(i / pi), i % pi, :, :] for i in range(pi * pi)]
    # f = [l[i].contiguous().view(-1, c * patch_size * patch_size) for i in range(pi * pi)]
    #
    # stack_tensor = tf.stack(f)
    #
    # stack_tensor = stack_tensor.permute(1, 0, 2)
    # return stack_tensor


class Linear():
    def __init__(self, name, in_feature, out_feature):
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.weight = tf.get_variable(name="weight_" + name, shape=[in_feature, out_feature], dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        self.bias = tf.get_variable(name="bias_" + name, shape=[out_feature], dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())

    def forward(self, input):
        return tf.matmul(input, self.weight) + self.bias


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
        kk3 = tf.nn.relu(self.g1.forward(p3))

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


class Identity_Regressor():
    def __init__(self, n, classes):
        self.fc1 = Linear("fc1", n, 256)
        self.fc2 = Linear("fc2", 256, classes)

    def forward(self, x):
        bs, m, n = x.size()
        x = tf.reshape(-1, n*m)
        x = tf.nn.relu(self.fc1.forward(x))
        x = self.fc2.forward(x)
        return x


if __name__ == '__main__':
    # [batch, in_height, in_width, in_channels]
    rd = tf.random_normal([10, 250, 250, 1], mean=-1, stddev=4)
    # import tensorflow as tf
    #
    # t = [1, 2, 3, 4, 5]
    # x = tf.strided_slice(t, [0], [5], [1])
    # y = tf.strided_slice(t, [1], [-2])
    # with tf.Session() as sess:
    #     print(sess.run(x))
    #     print(sess.run(y))

    sess = tf.Session()
    tf.global_variables_initializer()

    base_model = constractive_cnn.ConstractiveFourLayers(rd).forward()

    genarator_model = GenModel(512)

    ans = genarator_model.forward(base_model)

    print(sess.run(ans))


