# This note builds on tf_tutorial_mnist.py.
# On top of code inside tf_tutorial_mnist.py pasted here,
# Tensorboard functionalities are added.
# Name scopes are used to organize the output.
# Reference urls @ https://www.tensorflow.org/get_started/summaries_and_tensorboard
#                @ https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py

# Functions used:
# tf.summary.scalar()
# tf.summary.image()
# tf.summary.histogram()
# tf.summary.merge_all()
#

# Objects:
# tf.summary.FileWriter()
# FileWriter.add_summary()
# FileWriter.add_run_metadata() line 180

import tensorflow as tf
import time

def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
    
def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
    
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def variable_summaries(var):
    """Attach summaries to variables"""
    with tf.name_scope('summaries'):
        mean=tf.reduce_mean(var)
        tf.summary.scalar('mean',mean)
        with tf.name_scope('stddev'):
            stddev=tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev',stddev)
        tf.summary.scalar('max',tf.reduce_max(var))
        tf.summary.scalar('min',tf.reduce_min(var))
        tf.summary.histogram('histogram',var)
        
def feature_imgs(conv):
    """This function puts the conv outputs into images stacked vertically"""
    features=tf.unstack(conv,axis=3)
    conv_max=tf.reduce_max(conv)
    features_padded=map(lambda t: tf.pad(t-conv_max,[[0,0],[0,1],[0,0]])+conv_max,features)
    imgs=tf.expand_dims(tf.concat(features_padded,1),-1)
    return imgs
    
if __name__=='__main__':
    #load data
    from tensorflow.examples.tutorials.mnist import input_data
    print('Loading data...')
    mnist=input_data.read_data_sets('MNIST_data',one_hot=True)
    
    # Build the structure of the net
    # Data holders: input-output channels
    sess=tf.InteractiveSession()
    
    with tf.name_scope('input'):
        x=tf.placeholder(tf.float32,shape=[None, 784]) #28*28 pics
        y_=tf.placeholder(tf.float32, shape=[None,10]) #0-9 digit classes
    
    # first conv-pool layer
    # 32 5*5 kernels
    # 2*2 maxpool
    with tf.name_scope('layer_conv1'):
        with tf.name_scope('weights'):    
            W_conv1=weight_variable([5,5,1,32]) #[height,width,channels,num_feature_map]
            variable_summaries(W_conv1)
            tf.summary.image('kernel1',feature_imgs(W_conv1),10)
        with tf.name_scope('biases'):    
            b_conv1=bias_variable([32])
            variable_summaries(b_conv1)
    with tf.name_scope('input_reshape'):        
        x_image=tf.reshape(x,[-1,28,28,1]) #[batch, in_height,in_width,channels]    
        tf.summary.image('input',x_image,10) #
    
    with tf.name_scope('layer_conv1'):
        with tf.name_scope('relu'):
            h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
            tf.summary.image('relu',feature_imgs(h_conv1),10)
        with tf.name_scope('max_pool'):
            h_pool1=max_pool_2x2(h_conv1) #reduces the dim of pics down to 14*14
            tf.summary.image('max_pool',feature_imgs(h_pool1),10)

    # second conv-pool layer
    # 64 5*5 kernels
    # 2*2 maxpool
    with tf.name_scope('layer_conv2'):
        with tf.name_scope('weights'):
            W_conv2=weight_variable([5,5,32,64])
            variable_summaries(W_conv2)
            tf.summary.image('kernels2',feature_imgs(W_conv2),10)
        with tf.name_scope('biases'):
            b_conv2=bias_variable([64])
            variable_summaries(b_conv2)
        with tf.name_scope('relu'):
            h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
            tf.summary.image('relu',feature_imgs(h_conv2),10)
        with tf.name_scope('max_pool'):
            h_pool2=max_pool_2x2(h_conv2) # now the pics dim is 7*7
            tf.summary.image('max_pool',feature_imgs(h_pool2),10)
    
    # dense layer
    # 1024 nodes
    with tf.name_scope('layer_dense'):
        with tf.name_scope('weights'):
            W_fc1=weight_variable([7*7*64,1024])
            variable_summaries(W_fc1)
        with tf.name_scope('biases'):
            b_fc1=bias_variable([1024])
            variable_summaries(b_fc1) 
        with tf.name_scope('flat_input'):
            h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64]) #flaten the input
            tf.summary.histogram('flat_input',h_pool2_flat)
        with tf.name_scope('relu'):
            h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1)+b_fc1)
            tf.summary.histogram('h_fc1',h_fc1)
    
    #Dropout
    keep_prob=tf.placeholder(tf.float32) #holder for activation probability
    h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)
    
    #Output Layer
    with tf.name_scope("layer_out"):
        with tf.name_scope('weights'):
            W_fc2=weight_variable([1024,10])
            variable_summaries(W_fc2)
        with tf.name_scope('biases'):
            b_fc2=bias_variable([10])
            variable_summaries(b_fc2)
        with tf.name_scope('linear'):
            y_conv=tf.matmul(h_fc1_drop,W_fc2)+b_fc2
            tf.summary.histogram('linear',y_conv)
    #Now the architecture of the net is fully built.
    #To complete the computation graph, a loss needs to be added.
    
    #Loss is stacked on top of the graph
    with tf.name_scope('cross_entropy'):
        cross_entropy=tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_conv)
            )
        tf.summary.scalar('cross_entropy',cross_entropy)

    
    #An Optimizer is created
    with tf.name_scope('train'):
        train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction=tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
        with tf.name_scope('accuracy'):
            accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    tf.summary.scalar('accuracy',accuracy)
    
    merged=tf.summary.merge_all()
    train_writer=tf.summary.FileWriter('log/train',sess.graph)

    
    #initalize all variables
    sess.run(tf.global_variables_initializer()) #tf.global_variables_initializer().run()
    for i in range(1500):
        batch=mnist.train.next_batch(50) #an iterator provided by tf
        if i %100==0: #record excution stats and print accuracy
            run_options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata=tf.RunMetadata()
            summary,_=sess.run([merged,train_step],
                                feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5},
                                options=run_options,
                                run_metadata=run_metadata)
            train_writer.add_run_metadata(run_metadata,'step%03d'%i)
            train_writer.add_summary(summary,i)
            print('Adding run metadata for',i)
            train_accuracy=accuracy.eval(feed_dict={
                x:batch[0],y_:batch[1],keep_prob:1.0
            })
            print('step %d, training accuracy %g'%(i,train_accuracy))
        else:
            summary,_=sess.run([merged,train_step],feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})
            train_writer.add_summary(summary,i)
    train_writer.close()
    
    print('test accuracy %g'%accuracy.eval(feed_dict={
        x:mnist.test.images,y_:mnist.test.labels, keep_prob:1.0
    }))