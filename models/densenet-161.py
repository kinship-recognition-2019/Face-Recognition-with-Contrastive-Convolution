import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.framework import arg_scope

growth_k=48
def conv_layer(input, filter, kernel, stride=1, layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, use_bias=False, filters=filter, kernel_size=kernel, strides=stride, padding='SAME')
        return network

def Global_Average_Pooling(x, stride=1):


    return global_avg_pool(x, name='Global_avg_pooling')



def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True) :
        return tf.cond(training,
                       lambda : batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda : batch_norm(inputs=x, is_training=training, reuse=True))

def Drop_out(x, rate, training) :
    return tf.layers.dropout(inputs=x, rate=rate, training=training)

def Relu(x):
    return tf.nn.relu(x)

def Average_pooling(x, pool_size=[2,2], stride=2, padding='VALID'):
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def Max_Pooling(x, pool_size=[3,3], stride=2, padding='VALID'):
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Concatenation(layers) :
    return tf.concat(layers, axis=3)

def Linear(x,class_num=128) :
    return tf.layers.dense(inputs=x, units=class_num, name='linear')

def inference(images,keep_probability,phase_train=True,bottleneck_layer_size=128,weight_decay=0.0,reuse=None):
    return Dense_net(images,nb_block=2,filters=growth_k,is_training=phase_train,class_num=bottleneck_layer_size,dropout_rate=1-keep_probability)

def bottleneck_layer(x, scope,dropout_rate,is_training,filters=48):
    # print(x)
    with tf.name_scope(scope):
        x = Batch_Normalization(x, training=is_training, scope=scope+'_batch1')
        x = Relu(x)
        x = conv_layer(x, filter=4 * filters, kernel=[1,1], layer_name=scope+'_conv1')
        x = Drop_out(x, rate=dropout_rate, training=is_training)

        x = Batch_Normalization(x, training=is_training, scope=scope+'_batch2')
        x = Relu(x)
        x = conv_layer(x, filter=filters, kernel=[3,3], layer_name=scope+'_conv2')
        x = Drop_out(x, rate=dropout_rate, training=is_training)

        return x

def transition_layer( x, scope,is_training,dropout_rate):
    with tf.name_scope(scope):
        x = Batch_Normalization(x, training=is_training, scope=scope+'_batch1')
        x = Relu(x)
        # x = conv_layer(x, filter=self.filters, kernel=[1,1], layer_name=scope+'_conv1')
        

        
        #in_channel = x.shape[-1].as_list()
        #in_channel=list(in_channel)
        in_channel=x.shape.as_list()[-1]
        #print(type(in_channel))
        x = conv_layer(x, filter=in_channel*0.5, kernel=[1,1], layer_name=scope+'_conv1')
        x = Drop_out(x, rate=dropout_rate, training=is_training)
        x = Average_pooling(x, pool_size=[2,2], stride=2)

        return x

def dense_block( input_x, nb_layers, layer_name,filters,dropout_rate,is_training):
    with tf.name_scope(layer_name):
        layers_concat = list()
        layers_concat.append(input_x)

        x = bottleneck_layer(input_x, scope=layer_name + '_bottleN_' + str(0),filters=filters,dropout_rate=dropout_rate,is_training=is_training)

        layers_concat.append(x)

        for i in range(nb_layers - 1):
            x = Concatenation(layers_concat)
            x = bottleneck_layer(x, scope=layer_name + '_bottleN_' + str(i + 1),filters=filters,dropout_rate=dropout_rate,is_training=is_training)
            layers_concat.append(x)

        x = Concatenation(layers_concat)

        return x

def Dense_net(input_x,nb_block=2,filters=48,is_training=True,class_num=128,dropout_rate=0.2):
    x = conv_layer(input_x, filter=2 * filters, kernel=[7,7], stride=2, layer_name='conv0')
    x = Max_Pooling(x, pool_size=[3,3], stride=2)


    """
    for i in range(self.nb_blocks) :
        # 6 -> 12 -> 48
        x = self.dense_block(input_x=x, nb_layers=4, layer_name='dense_'+str(i))
        x = self.transition_layer(x, scope='trans_'+str(i))
    """




    x = dense_block(input_x=x, nb_layers=6, layer_name='dense_1',filters=filters,is_training=is_training,dropout_rate=dropout_rate)
    x = transition_layer(x, scope='trans_1',is_training=is_training,dropout_rate=dropout_rate)

    x = dense_block(input_x=x, nb_layers=12, layer_name='dense_2',filters=filters,is_training=is_training,dropout_rate=dropout_rate)
    x = transition_layer(x, scope='trans_2',is_training=is_training,dropout_rate=dropout_rate)

    x = dense_block(input_x=x, nb_layers=36, layer_name='dense_3',filters=filters,is_training=is_training,dropout_rate=dropout_rate)
    x = transition_layer(x, scope='trans_3',is_training=is_training,dropout_rate=dropout_rate)

    x = dense_block(input_x=x, nb_layers=24, layer_name='dense_final',filters=filters,is_training=is_training,dropout_rate=dropout_rate)



    x = Batch_Normalization(x, training=is_training, scope='linear_batch')
    x = Relu(x)
    x = Global_Average_Pooling(x)
    x = flatten(x)
    x = Linear(x,class_num=class_num)


    # x = tf.reshape(x, [-1, 10])
    return x,x
