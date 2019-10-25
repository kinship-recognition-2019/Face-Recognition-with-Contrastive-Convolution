import tensorflow as tf
import numpy as np
from contrastive_cnn import ConstractiveFourLayers
from CASIA_dataset import CasiaFaceDataset
from LFW_dataset import LFWDataset
from eval_metrics import evaluate
from gen_model import GenModel
from regressor import Regressor
from identity_regressor import IdentityRegressor
from conv_functions import group_conv_op
import argparse
import os, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

GLOBAL_BATCH_SIZE = 32


def compute_contrastive_features(data_1, data_2, basemodel, gen_model):

    data_1 = basemodel.forward(data_1, "data_1")
    data_2 = basemodel.forward(data_2, "data_2")
    # print(data_1) # ?*5*5*512

    kernel_1 = gen_model.forward(data_1, "kernel_1")
    kernel_2 = gen_model.forward(data_2, "kernel_2")
    # print(kernel_1) # ?*14*4608

    norm_kernel1 = tf.norm(kernel_1, 2, 2)
    norm_kernel2 = tf.norm(kernel_2, 2, 2)
    # print(norm_kernel1) # ?*14

    norm_kernel1_1 = tf.expand_dims(norm_kernel1, 2, name=None)
    norm_kernel2_2 = tf.expand_dims(norm_kernel2, 2, name=None)
    # print(norm_kernel1_1) # ?*14*1

    kernel_1 = kernel_1 / norm_kernel1_1
    kernel_2 = kernel_2 / norm_kernel2_2
    # print(kernel_1) # ?*14*4608

    F1, F2 = data_1, data_2
    Kab = tf.abs(kernel_1 - kernel_2)
    # print("Kab", Kab)  # ?*14*4608

    featuresdim, h, w = int(F1.shape[3]), int(F1.shape[1]), int(F1.shape[2])
    # print(featuresdim) # 512

    # F1 = tf.reshape(tensor=F1, shape=(-1, h, w, GLOBAL_BATCH_SIZE * featuresdim))
    # F2 = tf.reshape(tensor=F2, shape=(-1, h, w, GLOBAL_BATCH_SIZE * featuresdim))
    # print(F1.shape) # 1*5*5*16384

    noofkernels = 14
    kernelsize = 3

    # T = tf.reshape(Kab, (kernelsize, kernelsize, featuresdim * GLOBAL_BATCH_SIZE, noofkernels))
    T = tf.reshape(Kab, (noofkernels, kernelsize, kernelsize, 512, GLOBAL_BATCH_SIZE)) # 14*3*3*512*32

    # print("F1", F1)
    # print("T", T)

    F1_T_out = group_conv_op(input_op=F1, kernel=T, dh=1, dw=1) # 14*5*5*32
    F2_T_out = group_conv_op(input_op=F2, kernel=T, dh=1, dw=1)
    # print("F1_T_out", F1_T_out)

    # F1_T_out = tf.reshape(F1_T_out, (1, h, w, -1))
    # F2_T_out = tf.reshape(F2_T_out, (1, h, w, -1))
    # print(F1_T_out)

    # print(F1_T_out) # 1*5*5*896  pytorch=1*7*7*896
    # p, q, r, s = F1_T_out.size()

    A_list = tf.reshape(F1_T_out, (-1, 14*5*5*32))
    B_list = tf.reshape(F2_T_out, (-1, 14*5*5*32))

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
    testset = LFWDataset()

    base_model = ConstractiveFourLayers()
    gen_model = GenModel(512)
    reg_model = Regressor(350*32)
    idreg_model = IdentityRegressor(14*512*3*3, args.num_classes)

    input1 = tf.placeholder(tf.float32, [None, 128, 128, 1])
    input2 = tf.placeholder(tf.float32, [None, 128, 128, 1])
    target = tf.placeholder(tf.float32, [None, 1])
    c1 = tf.placeholder(tf.float32, [None, args.num_classes])
    c2 = tf.placeholder(tf.float32, [None, args.num_classes])

    A_list, B_list, org_kernel_1, org_kernel_2 = compute_contrastive_features(input1, input2, base_model, gen_model)

    reg_1 = reg_model.forward(A_list)
    reg_2 = reg_model.forward(B_list)

    SAB = tf.add(reg_1, reg_2) / 2.0

    hk1 = idreg_model.forward(org_kernel_1)
    hk2 = idreg_model.forward(org_kernel_2)
    # print("target", target)
    # print("SAB", SAB)

    loss1 = tf.losses.sigmoid_cross_entropy(multi_class_labels=target, logits=SAB)
    # cross_entropy1_1 = tf.reduce_mean(-tf.reduce_sum(target * tf.log(SAB), reduction_indices=[1]))
    # cross_entropy1_2 = tf.reduce_mean(-tf.reduce_sum(tf.subtract(tf.constant(1, dtype=tf.float32, shape=[GLOBAL_BATCH_SIZE, 1]), target) * tf.subtract(tf.constant(1, dtype=tf.float32, shape=[GLOBAL_BATCH_SIZE, 1]), tf.log(SAB)), reduction_indices=[1]))
    # loss1 = tf.add(cross_entropy1_1, cross_entropy1_2) * 0.5
    # loss2 = tf.losses.softmax_cross_entropy(onehot_labels=c1, logits=hk1) + tf.losses.softmax_cross_entropy(onehot_labels=c2, logits=hk2)
    cross_entropy_1 = tf.reduce_mean(-tf.reduce_sum(c1 * tf.log(hk1), reduction_indices=[1]))
    # cross_entropy_1 = tf.losses.softmax_cross_entropy(onehot_labels=c1, logits=hk1)
    # cross_entropy_2 = tf.losses.softmax_cross_entropy(onehot_labels=c2, logits=hk2)
    cross_entropy_2 = tf.reduce_mean(-tf.reduce_sum(c2 * tf.log(hk2), reduction_indices=[1]))
    loss2 = tf.add(cross_entropy_1, cross_entropy_2) * 0.5
    loss = tf.add(loss1, loss2)

    optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        f = open("result.txt", "w")
        for iteration in range(args.iters):

            data_1_batch, data_2_batch, c1_batch, c2_batch, target_batch = dataset.get_batch(batch_size=GLOBAL_BATCH_SIZE)
            # print(target_batch.shape)

            # data_1_cur, data_2_cur, c1_cur, c2_cur, target_cur = sess.run([data_1_batch, data_2_batch, c1_batch, c2_batch, target_batch])
            _, loss_val, loss1_val, loss2_val, reg1_val, reg2_val = sess.run([optimizer, loss, loss1, loss2, reg_1, reg_2],
                feed_dict={input1: data_1_batch, input2: data_2_batch, c1: c1_batch, c2: c2_batch, target: target_batch})

            print("Itera {0} : loss = {1}, loss1 = {2}, loss2 = {3}".format(iteration, loss_val, loss1_val, loss2_val))
            f.write("Itera {0} : loss = {1}, loss1 = {2}, loss2 = {3}\r\n".format(iteration, loss_val, loss1_val, loss2_val))
            f.flush()

            if(iteration != 0 and iteration % 100 == 0):
                acc_pool, start_time = [], time.time()
                for i in range(50):
                    test_1_batch, test_2_batch, label_batch = testset.get_batch(batch_size=GLOBAL_BATCH_SIZE)

                #     test_1_cur, test_2_cur, label_cur = sess.run([data_1_batch, data_2_batch, label_batch])
                    # out1_a, out1_b, k1, k2 = sess.run(compute_contrastive_features(test_1_batch, test_2_batch, base_model, gen_model))
                    SAB_val, reg1_val, reg2_val = sess.run([SAB, reg_1, reg_2], feed_dict={input1: test_1_batch, input2: test_2_batch})
                    # print("SAB", SAB_val)
                    # print("1v", reg1_val)
                    # print("2v", reg2_val)
                    dists = np.array(SAB_val).reshape((-1, 1))
                    # print(dists)
                    labels = np.array(label_batch).reshape((-1, 1))
                    # print(labels)
                    accuracy = evaluate(1.0 - dists, labels)

                    acc_pool.append(np.mean(accuracy))
                print("Acc(%.2f)"%(time.time()-start_time), np.mean(acc_pool), acc_pool)
                f.write("Acc" + str(np.mean(acc_pool)) + str(acc_pool) + str("\r\n"))
                f.flush()
        f.close()


if __name__ == '__main__':
    main()
    # data1 = tf.random_normal([1, 5, 5, 32768], mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
    # data1 = tf.random_normal([32, 128, 128, 1], mean=0.0,stddev=1.0,dtype=tf.float32,seed=None,name=None)
    # data2 = tf.random_normal([32, 128, 128, 1], mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
    #
    # sess = tf.Session()
    #
    #
    # base_model = ConstractiveFourLayers()
    # genarator_model = GenModel(512)
    # A_list, B_list, org_kernel_1, org_kernel_2 = compute_contrastive_features(data1, data2, base_model, genarator_model)
    # reg_model = Regressor(350) # pytorch = 686
    # print(A_list)
    # print(reg_model.forward(A_list))
    #
    # idreg_model = IdentityRegressor(14*512*3*3, 10575)
    # hk1 = idreg_model.forward(org_kernel_1)
    # reg_model.forward()

    # process = genarator_model.forward(base_model.forward(data1, "p"), "pp")

    # sess.run(tf.global_variables_initializer())
    # print(sess.run(ans))

    # print(ans.shape)

    # a = tf.constant([1, 2, 3])
    # print(a.shape)
    # print(tf.squeeze(a, 0))




