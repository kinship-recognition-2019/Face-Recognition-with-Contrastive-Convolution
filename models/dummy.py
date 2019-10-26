
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
  
def inference(images, keep_probability, phase_train=True, 
              bottleneck_layer_size=128, bottleneck_layer_activation=None, weight_decay=0.0, reuse=None):  
    batch_norm_params = {

        'decay': 0.995,

        'epsilon': 0.001,

        'updates_collections': None,

        'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
    }
    
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        size = np.prod(images.get_shape()[1:].as_list())
        net = slim.fully_connected(tf.reshape(images, (-1,size)), bottleneck_layer_size, activation_fn=None, 
                scope='Bottleneck', reuse=False)
        return net, None
