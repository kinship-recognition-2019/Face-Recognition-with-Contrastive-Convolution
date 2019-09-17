import tensorflow as tf
import numpy as np
from contrastive_cnn import ConstractiveFourLayers
from CASIA_dataset import CasiaFaceDataset
from IFW_dataset import IFWDataset
from eval_metrics import evaluate
from linear import Linear
from gen_model import GenModel
from regressor import Regressor
from identity_regressor import IdentityRegressor
import argparse
import os, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

GLOBAL_BATCH_SIZE = 32


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
    # print(kernel_1) # 64 * 14 * 4608

    F1, F2 = data_1, data_2
    Kab = tf.abs(kernel_1 - kernel_2)

    # bs, featuresdim, h, w = int(), int(F1.shape[3]), int(F1.shape[1]), int(F1.shape[2])
    featuresdim, h, w = int(F1.shape[3]), int(F1.shape[1]), int(F1.shape[2])

    # F1 = tf.reshape(tensor=F1, shape=(1, h, w, bs * featuresdim))
    # F2 = tf.reshape(tensor=F2, shape=(1, h, w, bs * featuresdim))
    F1 = tf.reshape(tensor=F1, shape=(-1, h, w, GLOBAL_BATCH_SIZE * featuresdim))
    F2 = tf.reshape(tensor=F2, shape=(-1, h, w, GLOBAL_BATCH_SIZE * featuresdim))
    # print(F1.shape) # 1*5*5*32768

    # noofkernels = 14
    # kernelsize = 3
    # T = tf.reshape(Kab, (-1, kernelsize, kernelsize, 32768))

    kernel = tf.get_variable(name="kernel", shape=[3, 3, GLOBAL_BATCH_SIZE * featuresdim, 14*GLOBAL_BATCH_SIZE], dtype=tf.float32,
                         initializer=tf.contrib.layers.xavier_initializer_conv2d())
    # kernel = tf.get_variable(name="kernel", dtype=tf.float32, initializer=T)

    F1_T_out = tf.nn.conv2d(input=F1, filter=kernel, strides=[1, 1, 1, 1], padding='SAME')
    F2_T_out = tf.nn.conv2d(input=F2, filter=kernel, strides=[1, 1, 1, 1], padding='SAME')

    # print(F1_T_out) # 1*5*5*896  pytorch=1*7*7*896
    # p, q, r, s = F1_T_out.size()

    A_list = tf.reshape(F1_T_out, (-1, 350))
    B_list = tf.reshape(F2_T_out, (-1, 350))

    # print(A_list) # 64*350  pytorch=64*686

    return A_list, B_list, kernel_1, kernel_2


def main():
    parser = argparse.ArgumentParser(description="Tensorflow Contrastive Convolution")
    parser.add_argument('--num_classes', default=10575, type=int,
                        metavar='N', help='number of classes (default: 5749)')
    parser.add_argument('--iters', type=int, default = 200000, metavar='N',
                        help='number of iterations to train (default: 10)')
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
    # loss1 = tf.reduce_sum(tf.square(tf.substract(target, SAB)))
    cross_entropy1 = tf.reduce_mean(-tf.reduce_sum(c1 * tf.log(hk1), reduction_indices=[1]))
    cross_entropy2 = tf.reduce_mean(-tf.reduce_sum(c2 * tf.log(hk2), reduction_indices=[1]))
    loss2 = tf.add(cross_entropy1, cross_entropy2) * 0.5
    loss = tf.add(loss1, loss2)

    optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for iteration in range(args.iters):

            data_1_batch, data_2_batch, c1_batch, c2_batch, target_batch = dataset.get_batch(batch_size=GLOBAL_BATCH_SIZE)

            # data_1_cur, data_2_cur, c1_cur, c2_cur, target_cur = sess.run([data_1_batch, data_2_batch, c1_batch, c2_batch, target_batch])
            _, loss_val = sess.run([optimizer, loss], feed_dict={input1: data_1_batch, input2: data_2_batch, c1: c1_batch, c2: c2_batch, target: target_batch})
            # print(iteration, time.time()-start_time, loss_val)
            print("Itera {0} : {1}".format(iteration, loss_val))

            if(iteration % 20 == 0):
                acc_pool, start_time = [], time.time()
                for i in range(10):
                    test_1_batch, test_2_batch, label_batch = testset.get_batch(batch_size=GLOBAL_BATCH_SIZE)

                #     test_1_cur, test_2_cur, label_cur = sess.run([data_1_batch, data_2_batch, label_batch])
                    # out1_a, out1_b, k1, k2 = sess.run(compute_contrastive_features(test_1_batch, test_2_batch, base_model, gen_model))
                    SAB_val  = sess.run([SAB], feed_dict={input1: test_1_batch, input2: test_2_batch})
                    # print(SAB_val)
                #
                    dists = np.array(SAB_val).reshape((-1, 1))
                    labels = np.array(label_batch)
                    accuracy = evaluate(1.0 - dists, labels)
                    acc_pool.append(np.mean(accuracy))
                print("Acc(%.2f)"%(time.time()-start_time), np.mean(acc_pool))


if __name__ == '__main__':
    main()
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




