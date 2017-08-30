import numpy as np
import tensorflow as tf
import os
import time
from get_database import Dataset as MRDB
from get_database import plot
from Unet import net as model

import ipdb

# User parameter
name_of_exp   = "Unet_LR_resid"
isResidual    = True
nBatch        = 10
nEpoch        = 200


# load MRDB
db            = MRDB("/home/user/data/mat","/KNEE_DS6LR.mat")
totalN        = db.totalN

train_size    = len(db.img_train)
valid_size    = len(db.img_valid)
[nY, nX, nCh] = db.img_shape
dtype         = tf.float32

# check the directory
exp_dir  = "./result/"+name_of_exp
ckpt_dir = "./result/"+name_of_exp+"/ckpt_dir"
log_dir  = "./result/"+name_of_exp+"/log_dir"
if not os.path.exists(exp_dir):
    os.mkdir(exp_dir)
if not os.path.exists(ckpt_dir):
    os.mkdir(ckpt_dir)
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

# function needed on tf
def tf_ssos( x_ri ):
    return tf.sqrt( tf.reduce_sum( tf.square( tf.slice(x_ri, [0,0,0,0], [-1,-1,-1,nCh])) + tf.square( tf.slice( x_ri, [0,0,0,nCh],[-1,-1,-1,nCh])), axis=3, keep_dims=True) )


with tf.device("/:gpu1"):
    # Data feed dict and network
    input_node   = tf.placeholder(dtype, shape = (None, nY, nX, nCh*2) ) # nCH*2 to convert complex channel to real/imag
    target_node  = tf.placeholder(dtype, shape = (None, nY, nX, nCh*2) )

    net_out      = model(input_node, isResidual=isResidual)

    loss         = tf.losses.mean_squared_error(labels=target_node, predictions=net_out, weights=1.0)
    tf.summary.scalar("objective function(loss)",loss) 
    target_ssos  = tf_ssos( target_node )
    input_ssos   = tf_ssos( input_node )
    netout_ssos  = tf_ssos( net_out )

    tf.summary.image("target ssos", target_ssos)
    tf.summary.image("input_ssos", input_ssos)
    tf.summary.image("netout_ssos", netout_ssos)

    NMSE         = tf.losses.mean_squared_error( labels=target_ssos, predictions=netout_ssos )
    RNMSE        = tf.sqrt( NMSE )
    tf.summary.scalar("NMSE", NMSE)
    tf.summary.scalar("RNMSE", RNMSE)

    PIXEL_MAX     = 256.0
    _20_div_Log10 =8.6859
    PSNR          = tf.log(PIXEL_MAX/tf.sqrt(NMSE))*_20_div_Log10
    tf.summary.scalar("PSNR",PSNR)

    batch        = tf.Variable(0, dtype=dtype)
    lr           = tf.train.exponential_decay(learning_rate=0.01, global_step= batch*nBatch, decay_steps=totalN, decay_rate=0.95, staircase=False)

    optimizer    = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5, beta2=0.9).minimize(loss, colocate_gradients_with_ops=True)

    merged_all   = tf.summary.merge_all()
saver        = tf.train.Saver()

def myNumExtractor(s):
    head = s.rstrip('0123456789')
    tail = s[len(head):]
    return int(tail)

# Main loop goes here

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    len_batch     = int(train_size/nBatch)
    len_batch_val = int(valid_size/nBatch)
    
    #check wheter it have been trained or not
    latest_ckpt = tf.train.latest_checkpoint(ckpt_dir)
    if latest_ckpt==None:
        print("Start! initially!")
        tf.global_variables_initializer().run()
        epoch_start=0
    else:
        print("Start from save model -"+latest_ckpt)
        saver.restore(sess, latest_ckpt)
        epoch_start=myNumExtractor(latest_ckpt)+1

    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

    for iEpoch in range(epoch_start, nEpoch):
        start_time = time.time()

        #Loop over all batches
        db.shuffleTrainIDs()
        loss_train = 0.0

        for iBatch in range(len_batch):
            offset = (iBatch*nBatch)
            # takes about 1 sec for getBatch
            input_batch, target_batch = db.getBatch("train", offset, offset+nBatch, doAug=0)
            _, merged, l_train = sess.run([optimizer, merged_all, loss], feed_dict = {input_node:input_batch, target_node:target_batch})

            if (iBatch%20)==1:
                print("---------processing EPOCH#%d : LOSS %.4f" %(iEpoch, l_train))

            loss_train += l_train
            summary_writer.add_summary(merged, iEpoch*len_batch+iBatch)

        print("EPOCH(%d-train)--Loss : %.4f" %(iEpoch, loss_train/len_batch))

        loss_valid = 0.0
        for iBatch_val in range(len_batch_val):
            offset     = iBatch_val*nBatch
            input_batch, target_batch = db.getBatch("valid", offset, offset+nBatch, doAug=0)
            
            l_valid, prediction_valid  = sess.run([loss, net_out], feed_dict={input_node:input_batch, target_node:target_batch})
            loss_valid +=l_valid

        print("EPOCH(%d-valid)--Loss : %.4f" %(iEpoch, loss_valid/len_batch_val))

        print("TOTAL time for 1 epoch : %.2f min" % (float(time.time()-start_time)/60.0) )

        path_saved = saver.save(sess, os.path.join(ckpt_dir, "model.ckpt"), global_step=iEpoch)
        print("The model saved in file:"+path_saved)




