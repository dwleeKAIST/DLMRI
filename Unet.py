import tensorflow as tf
import tensorflow.contrib.layers as li
import ipdb

def conv2d(x, ch_out,name):
    return tf.layers.conv2d(x,filters=ch_out, kernel_size=(3,3), strides=(1,1), padding="SAME", use_bias=False, kernel_initializer=li.xavier_initializer(), kernel_regularizer=None,name=name)
def pool2d(x, ch_out, name):
    return tf.layers.conv2d(x, filters=ch_out, kernel_size=(2,2), strides=(2,2), padding="SAME", use_bias=False, kernel_initializer=li.xavier_initializer(), kernel_regularizer=None, name=name)
def BN(x):
    return li.batch_norm( x, decay=0.999, center=True, scale=False, epsilon=0.001, activation_fn=None, param_initializers=None)
def conv2dT(x, ch_out, name):
    return tf.layers.conv2d_transpose(x, filters=ch_out, kernel_size=(2,2), strides=(2,2), kernel_initializer= li.xavier_initializer(), kernel_regularizer=None,name=name)
def ReLU(x):
    return tf.nn.relu( x )


# Define the network
def net(input_node, N_features=64, isResidual=False):
    nCh_in   = input_node.shape[3]
    
    # stage-0 
    stg0_1      = ReLU(   BN(   conv2d( input_node,  N_features, "stg0_1" ) ) )
    stg0_2      = ReLU(   BN(   conv2d(     stg0_1,  N_features, "stg0_2" ) ) )

    # stage-1
    stg1_pool   = pool2d( stg0_2,  N_features*2, "stg1_pool")
    stg1_1      = ReLU(   BN(   conv2d( stg1_pool,     N_features*2, "stg1_1" ) ) )
    stg1_2      = ReLU(   BN(   conv2d(    stg1_1,     N_features*2, "stg1_2" ) ) )

    # stage-2
    stg2_pool   = pool2d( stg1_2,  N_features*4, "stg2_pool")
    stg2_1      = ReLU(   BN(   conv2d( stg2_pool,    N_features*4, "stg2_1" ) ) )
    stg2_2      = ReLU(   BN(   conv2d(    stg2_1,    N_features*4, "stg2_2" ) ) )

    # stage-3
    stg3_pool   = pool2d( stg2_2, N_features*8, "stg3_pool")
    stg3_1      = ReLU(   BN(   conv2d( stg3_pool,  N_features*8, "stg3_1" ) ) )
    stg3_2      = ReLU(   BN(   conv2d(    stg3_1,  N_features*8, "stg3_2" ) ) )

    # stage-4
    stg4_pool   = pool2d( stg3_2, N_features*16, "stg4_pool")
    stg4_1      = ReLU(   BN(   conv2d( stg4_pool,  N_features*16, "stg4_1" ) ) )
    stg4_2      = ReLU(   BN(   conv2d(    stg4_1,  N_features*16, "stg4_2" ) ) )

    # stage3_back
    _stg3_cnvt  = conv2dT(    stg4_2,  ch_out=N_features*8, name= "stg3_up")
    _stg3_cnct  = tf.concat( [_stg3_cnvt, stg3_2], 3, "skip_conn_stg3") 
    _stg3_1     = ReLU(   BN(   conv2d( _stg3_cnct, N_features*8,  "stg3_1_up") ) )
    _stg3_2     = ReLU(   BN(   conv2d(    _stg3_1, N_features*8,  "stg3_2_up") ) )

    # stage2_back
    _stg2_cnvt  = conv2dT(    stg3_2,  ch_out=N_features*4, name= "stg2_up")
    _stg2_cnct  = tf.concat( [_stg2_cnvt, stg2_2], 3, "skip_conn_stg2") 
    _stg2_1     = ReLU(   BN(   conv2d( _stg2_cnct,  N_features*4,  "stg2_1_up") ) )
    _stg2_2     = ReLU(   BN(   conv2d(    _stg2_1,  N_features*4,  "stg2_2_up") ) )

    # stage1_back
    _stg1_cnvt  = conv2dT(    stg2_2,  ch_out=N_features*2, name= "stg1_up")
    _stg1_cnct  = tf.concat( [_stg1_cnvt, stg1_2], 3, "skip_conn_stg1") 
    _stg1_1     = ReLU(   BN(   conv2d( _stg1_cnct,  N_features*2,  "stg1_1_up") ) )
    _stg1_2     = ReLU(   BN(   conv2d(    _stg1_1,  N_features*2,  "stg1_2_up") ) )

    # stage0_back
    _stg0_cnvt  = conv2dT(    stg1_2, ch_out=N_features, name= "stg0_up")
    _stg0_cnct  = tf.concat( [_stg0_cnvt, stg0_2], 3, "skip_conn_stg0") 
    _stg0_1     = ReLU(   BN(   conv2d( _stg0_cnct,  N_features,  "stg0_1_up") ) )
    _stg0_2     = ReLU(   BN(   conv2d(    _stg0_1,  N_features,  "stg0_2_up") ) )

    if isResidual:
        out     = conv2d(  _stg0_2, nCh_in, "conv1x1") + input_node
        return out
    else:
        out     = conv2d(  _stg0_2, nCh_in, "conv1x1")
        return out










