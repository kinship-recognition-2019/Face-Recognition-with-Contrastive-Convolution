import tensorflow as tf
import numpy as np
from contrastive_cnn import ConstractiveFourLayers
from CASIA_dataset import CasiaFaceDataset
from IFW_dataset import IFWDataset
from eval_metrics import evaluate
import argparse


def extract_patches(x, patch_size):
    cur = tf.extract_image_patches(images=x, ksizes=[1, patch_size, patch_size, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding="VALID")

    dim, channel = int(cur.shape[1]) * int(cur.shape[2]), int(x.shape[3])
    cur = tf.reshape(cur, (-1, dim, channel*patch_size*patch_size))
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
        # print("Error", input, self.weight)

        return tf.add(tf.tensordot(input, self.weight, [[len(input.shape)-1], [0]]), self.bias)
        # else:
        #     return tf.add(tf.tensordot(input, self.weight, [[1], [0]]), self.bias)


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
        # bs, c = x.shape[0], x.shape[1]
        # print(x) # 64*350 pytorch 64*686
        x = self.linear.forward(x)
        # print(x) # 64*1
        x = tf.sigmoid(x)
        # print(x) # 64*1
        return x


class IdentityRegressor():
    def __init__(self, n, classes):
        self.fc1 = Linear("fc1", n, 256)
        self.fc2 = Linear("fc2", 256, classes)

    def forward(self, x):
        # bs, m, n = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
        # print("x", x)
        m, n = int(x.shape[1]), int(x.shape[2])

        # print(x) # 64*14*4608
        x = tf.reshape(x, (-1, n*m))
        # print(x) # 64*64512
        x = tf.nn.relu(self.fc1.forward(x))
        # print(x) # 64*256
        x = self.fc2.forward(x)
        x = tf.nn.softmax(x)
        # print(x) # 64*10575
        return x


def compute_contrastive_features(data_1, data_2, basemodel, gen_model):

    data_1 = basemodel.forward(data_1, "data_1")
    data_2 = basemodel.forward(data_2, "data_2")
    # print(data_1) # 64*5*5*512

    kernel_1 = gen_model.forward(data_1, "kernel_1")
    kernel_2 = gen_model.forward(data_2, "kernel_2")
    # print(kernel_1) # 64*14*4608

    norm_kernel1 = tf.norm(kernel_1, 2, 2)
    norm_kernel2 = tf.norm(kernel_2, 2, 2)
    # print(norm_kernel1) # 64*14

    norm_kernel1_1 = tf.expand_dims(norm_kernel1, 2, name=None)
    norm_kernel2_2 = tf.expand_dims(norm_kernel2, 2, name=None)
    # print(norm_kernel1_1) # 64*14*1

    kernel_1 = kernel_1 / norm_kernel1_1
    kernel_2 = kernel_2 / norm_kernel2_2
    # print("l", kernel_1) # 64 * 14 * 4608

    F1, F2 = data_1, data_2
    Kab = tf.abs(kernel_1 - kernel_2)
    # print(F1.shape)

    # bs, featuresdim, h, w = int(F1.shape[0]), int(F1.shape[3]), int(F1.shape[1]), int(F1.shape[2])
    featuresdim, h, w = int(F1.shape[3]), int(F1.shape[1]), int(F1.shape[2])
    # print("fhw")
    # print(featuresdim, h, w)
    # F1 = tf.reshape(tensor=F1, shape=(1, h, w, bs * featuresdim))
    # F2 = tf.reshape(tensor=F2, shape=(1, h, w, bs * featuresdim))
    F1 = tf.reshape(tensor=F1, shape=(-1, h, w, 64 * featuresdim))
    F2 = tf.reshape(tensor=F2, shape=(-1, h, w, 64 * featuresdim))
    # print(F1.shape) # 1*5*5*32768

    # noofkernels = 14
    # kernelsize = 3
    # T = tf.reshape(Kab, (-1, kernelsize, kernelsize, 32768))

    kernel = tf.get_variable(name="kernel", shape=[3, 3, 32768, 896], dtype=tf.float32,
                             initializer=tf.contrib.layers.xavier_initializer_conv2d())
    # kernel = tf.get_variable(name="kernel", dtype=tf.float32, initializer=T)

    F1_T_out = tf.nn.conv2d(input=F1, filter=kernel, strides=[1, 1, 1, 1], padding='SAME')
    F2_T_out = tf.nn.conv2d(input=F2, filter=kernel, strides=[1, 1, 1, 1], padding='SAME')

    # print(F1_T_out) # 1*5*5*896  pytorch=1*7*7*896
    # p, q, r, s = F1_T_out.size()

    A_list = tf.reshape(F1_T_out, (-1, 350))
    B_list = tf.reshape(F2_T_out, (-1, 350))
    # print(A_list.shape())
    # A_list = F1_T_out
    # B_list = F2_T_out
    # print(F1_T_out)
    # print(A_list) # 64*350  pytorch=64*686

    return A_list, B_list, kernel_1, kernel_2


def main():
    parser = argparse.ArgumentParser(description="Tensorflow Contrastive Convolution")
    parser.add_argument('--batch_size', type=int, default = 64 , metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--num_classes', default=10575, type=int,
                        metavar='N', help='number of classes (default: 5749)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--iters', type=int, default = 200000, metavar='N',
                        help='number of iterations to train (default: 10)')
    parser.add_argument('--epochs', type=int, default = 80, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lfw-dir', type=str,
                        default='dataset/lfw',
                        help='path to dataset')
    parser.add_argument('--lfw_pairs_path', type=str, default='dataset/lfw_pairs.txt',
                        help='path to pairs file')
    args = parser.parse_args()

    dataset = CasiaFaceDataset()
    testset = IFWDataset('dataset/lfw', 'dataset/pairs.txt')

    base_model = ConstractiveFourLayers()
    gen_model = GenModel(512)
    reg_model = Regressor(350)
    idreg_model = IdentityRegressor(14 * 512 * 3 * 3, args.num_classes)

    input1 = tf.placeholder(tf.float32, [None, 128, 128, 1])
    input2 = tf.placeholder(tf.float32, [None, 128, 128, 1])
    target = tf.placeholder(tf.float32, [None])
    c1 = tf.placeholder(tf.float32, [None, args.num_classes])
    c2 = tf.placeholder(tf.float32, [None, args.num_classes])

    A_list, B_list, org_kernel_1, org_kernel_2 = compute_contrastive_features(input1, input2, base_model, gen_model)

    reg_1 = reg_model.forward(A_list)
    reg_2 = reg_model.forward(B_list)

    SAB = tf.add(reg_1, reg_2) / 2.0

    hk1 = idreg_model.forward(org_kernel_1)
    hk2 = idreg_model.forward(org_kernel_2)

    loss1 = tf.reduce_mean(tf.abs(tf.subtract(target, SAB)))
    cross_entropy1 = tf.reduce_mean(-tf.reduce_sum(c1 * tf.log(hk1), reduction_indices=[1]))
    cross_entropy2 = tf.reduce_mean(-tf.reduce_sum(c2 * tf.log(hk2), reduction_indices=[1]))
    loss2 = tf.add(cross_entropy1, cross_entropy2) * 0.5
    loss = tf.add(loss1, loss2)

    optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for iteration in range(args.iters):
            data_1_batch, data_2_batch, c1_batch, c2_batch, target_batch = dataset.get_batch()

            data_1_cur, data_2_cur, c1_cur, c2_cur, target_cur = sess.run([data_1_batch, data_2_batch, c1_batch, c2_batch, target_batch])

            _, loss_val = sess.run([optimizer, loss], feed_dict={input1: data_1_cur, input2: data_2_cur, c1: c1_cur, c2: c2_cur, target: target_cur})
            print(iteration, loss_val)

            if(iteration % 1 == 0):
                test_1_batch, test_2_batch, label_batch = testset.get_batch()
                test_1_cur, test_2_cur, label_cur = sess.run([data_1_batch, data_2_batch, label_batch])
                out1_a, out1_b, k1, k2 = compute_contrastive_features(test_1_cur, test_2_cur, base_model, gen_model)

                SA = reg_model.forward(out1_a)
                SB = reg_model.forward(out1_b)
                SAB = tf.add(SA, SB) / 2.0
                SAB = tf.squeeze(SAB)

                dists = SAB.eval()
                labels = np.array(label_cur)

                dists = dists.eval()
                labels = np.array(labels)
                accuracy = evaluate(1 - dists, labels)


if __name__ == '__main__':
    # data1 = tf.random_normal([1, 5, 5, 32768], mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
    # data1 = tf.random_normal([64, 128, 128, 1], mean=0.0,stddev=1.0,dtype=tf.float32,seed=None,name=None)
    # data2 = tf.random_normal([64, 128, 128, 1], mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
    #
    # sess = tf.Session()
    #
    #
    # base_model = ConstractiveFourLayers()
    # genarator_model = GenModel(512)
    # A_list, B_list, org_kernel_1, org_kernel_2 = compute_contrastive_features(data1, data2, base_model, genarator_model,
    #                                                                           "s")
    # reg_model = Regressor(350) # pytorch = 686
    # reg_model.forward(A_list)
    #
    # idreg_model = IdentityRegressor(14*512*3*3, 10575)
    # hk1 = idreg_model.forward(org_kernel_1)
    # reg_model.forward()

    # process = genarator_model.forward(base_model.forward(data1, "p"), "pp")

    # sess.run(tf.global_variables_initializer())
    # print(sess.run(ans))

    # print(ans.shape)
    main()



