# Tensorflow mandates these.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import functools

import tensorflow as tf

slim = tf.contrib.slim


def ShuffleNet_v2(images, bottleneck_layer_size, is_training, reuse=False, shuffle=True, base_ch=24, groups=1):
    with tf.variable_scope('Stage1'):
        net = slim.conv2d(images, 24, [3, 3], 2)
        net = slim.max_pool2d(net, [3, 3], 2, padding='SAME')

    net = shuffle_stage(net, 176, 3, groups, is_training, 'Stage2')
    net = shuffle_stage(net, 352, 7, groups, is_training, 'Stage3')
    net = shuffle_stage(net, 704, 3, groups, is_training, 'Stage4')

    with tf.variable_scope('Stage5'):
        net = slim.dropout(net, 0.5, is_training=is_training)
        net = slim.conv2d(net, bottleneck_layer_size, [1, 1], stride=1,
                          padding="SAME",
                          scope="conv_stage5")
        net = slim.avg_pool2d(net, kernel_size=4, stride=1)
        net = tf.reduce_mean(net, [1, 2],  name="logits")

    return net, None


def shuffle_stage(net, output, repeat, group, is_training, scope="Stage"):
    with tf.variable_scope(scope):
        net = shuffle_bottleneck(net, output, 2, is_training, group, scope='Unit{}'.format(0))
        for i in range(repeat):
            net = shuffle_bottleneck(net, output, 1, is_training, group, scope='Unit{}'.format(i+1))
    return net


def shuffle_bottleneck(net, output, stride, is_training, group=1, scope="Unit"):
    if stride != 1:
        _b, _h, _w, _c = net.get_shape().as_list()
        output = output - _c

    with tf.variable_scope(scope):
        if stride != 1:
            net_skip = net
            with tf.variable_scope("3x3DXConvL"):
                depthwise_filter = tf.get_variable("depth_conv_w", [3, 3, _c, 1],
                                                   initializer=tf.truncated_normal_initializer(stddev=0.01))
                net_skip = tf.nn.depthwise_conv2d(net_skip, depthwise_filter, [1, stride, stride, 1], 'SAME', name="DWConvL")
                net_skip = slim.conv2d(net_skip, _c, [1, 1], 1)
        else:
            net_skip, net = channel_split(net)
            output = output/2

        net = slim.conv2d(net, output, [1, 1], 1)

        with tf.variable_scope("3x3DXConvR"):
            depthwise_filter = tf.get_variable("depth_conv_w", [3, 3, output, 1],
                                               initializer=tf.truncated_normal_initializer(stddev=0.01))
            net = tf.nn.depthwise_conv2d(net, depthwise_filter, [1, stride, stride, 1], 'SAME', name="DWConvR")

        net = slim.conv2d(net, output, [1, 1], 1)

        net = tf.concat([net_skip, net], axis=3)

        out_temp = net.get_shape()[3]
        net = channel_shuffle(net, out_temp, 8, scope="ChannelShuffle")

        net = tf.nn.relu(net)

    return net


def channel_split(inputs, num_splits=2):
    c = inputs.get_shape()[3]
    input1, input2 = tf.split(inputs, [int(c//num_splits), int(c//num_splits)], axis=3)
    return input1, input2


def channel_shuffle(net, output, group, scope="ChannelShuffle"):
    num_channels_in_group = output//group
    with tf.variable_scope(scope):
        net = tf.split(net, output, axis=3, name="split")
        chs = []
        for i in range(group):
            for j in range(num_channels_in_group):
                chs.append(net[i + j * group])
        net = tf.concat(chs, axis=3, name="concat")
    return net


def shufflenet_arg_scope(is_training=True,
                           weight_decay=0.00005,
                           regularize_depthwise=False):

  batch_norm_params = {
      'is_training': is_training,
      'center': True,
      'scale': True,
      'fused': True,
      'decay': 0.995,
      'epsilon': 2e-5,
      # force in-place updates of mean and variance estimates
      'updates_collections': None,
      # Moving averages ends up in the trainable variables collection
      'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
  }

  # Set weight_decay for weights in Conv and InvResBlock layers.
  #weights_init = tf.truncated_normal_initializer(stddev=stddev)
  weights_init = tf.contrib.layers.xavier_initializer(uniform=False)
  regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
  if regularize_depthwise:
    depthwise_regularizer = regularizer
  else:
    depthwise_regularizer = None
  with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                      weights_initializer=weights_init,
                      activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm): #tf.keras.layers.PReLU
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      with slim.arg_scope([slim.conv2d], weights_regularizer=regularizer):
        with slim.arg_scope([slim.separable_conv2d],
                            weights_regularizer=depthwise_regularizer) as sc:
          return sc


def inference(images, keep_probability,bottleneck_layer_size=128, phase_train=False,
              weight_decay=0.00005, reuse=False):

    # pdb.set_trace()
    arg_scope = shufflenet_arg_scope(is_training=phase_train, weight_decay=weight_decay)
    with slim.arg_scope(arg_scope):
        return ShuffleNet_v2(images, bottleneck_layer_size=bottleneck_layer_size, is_training=phase_train, reuse=reuse)
