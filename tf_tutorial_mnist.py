#####Notes Taken from Tensorflow Tutorial#####

#Tutorial @ url:https://www.tensorflow.org/get_started/mnist/pros

#This program builds a conv net with the following configurations:
#   input --> conv1--> pool1-->conv2-->pool2-->dense-->output-->xent_loss
#   input(mnist dataset): 28*28 imgs
#   conv1: 32 5*5 kernels, 1 stride, padded
#   pool1: 2*2 max_pool, output 14*14 feature_maps
#   conv2: 64 5x5 kernels, 1 stride, padded
#   pool2: 2*2 max_pool, output 7*7 feature_maps
#   dense: 1024 nodes, 0.5 dropout applied
#   output:linear output to 10 classes
#   xent_loss: cross_entropy_loss

##Functions Used:
#tf.truncated_normal(shape,stddev=)
#tf.nn.conv2d(x,W,strides=[],padding='')
#   x.shape=[batch,in_height,in_width,in_channels]
#   W.shape=[filter_height,filter_width,in_channels,out_channels('num_feature_maps')]   
#   strides=[batch_stride,height_stride,width_stride,channel_stride]
#   padding="SAME" or "VALID"
#tf.nn.max_pool(x,ksize=[],strides=[],padding='')
#   ksize.shape=[batch_windowsize,height_windowsize,width_windowsize,channel_windowsize]
#tf.nn.softmax_cross_entropy_logits(labels=,logits=)
#tf.nn.dropout(layer, activation_probability)
#tf.nn.relu()
#tf.argmax(matrix,axis=): returns a vector of indices of the largest items along axis=
#tf.equal()
#tfgraph.eval(feed_dict={})
#tfgraph.run(feed_dict={})
#tf.reshape(tensor,shape)
#tf.matmul(): matrix multipliaction
#tf.reduce_mean()
#tf.equal(ts1,ts2)

## Objects Used:
#tf.train.AdamOptimizer(learning_rate=).minimize(tfgraph)

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

if __name__=='__main__':
    #load data
    from tensorflow.examples.tutorials.mnist import input_data
    print('Loading data...')
    mnist=input_data.read_data_sets('MNIST_data',one_hot=True)
    
    # Build the structure of the net
    # Data holders: input-output channels
    x=tf.placeholder(tf.float32,shape=[None, 784]) #28*28 pics
    y_=tf.placeholder(tf.float32, shape=[None,10]) #0-9 digit classes
    
    # first conv-pool layer
    # 32 5*5 kernels
    # 2*2 maxpool
    W_conv1=weight_variable([5,5,1,32]) #[height,width,channels,num_feature_map]
    b_conv1=bias_variable([32])
    
    x_image=tf.reshape(x,[-1,28,28,1]) #[batch, in_height,in_width,channels]
    h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
    h_pool1=max_pool_2x2(h_conv1) #reduces the dim of pics down to 14*14
    
    # second conv-pool layer
    # 64 5*5 kernels
    # 2*2 maxpool
    W_conv2=weight_variable([5,5,32,64])
    b_conv2=bias_variable([64])
    h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
    h_pool2=max_pool_2x2(h_conv2) # now the pics dim is 7*7
    
    # dense layer
    # 1024 nodes
    W_fc1=weight_variable([7*7*64,1024])
    b_fc1=bias_variable([1024]) 
    
    h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64]) #flaten the input
    h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1)+b_fc1)
    
    #Dropout
    keep_prob=tf.placeholder(tf.float32) #holder for activation probability
    h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)
    
    #Output Layer
    W_fc2=weight_variable([1024,10])
    b_fc2=bias_variable([10])
    
    y_conv=tf.matmul(h_fc1_drop,W_fc2)+b_fc2
    #Now the architecture of the net is fully built.
    #To complete the computation graph, a loss needs to be added.
    
    #Loss is stacked on top of the graph
    cross_entropy=tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_conv)
    )
    
    #An Optimizer is created
    train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    
    correct_prediction=tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    
    with tf.Session() as sess:
        #initalize all variables
        sess.run(tf.global_variables_initializer())
        t=0
        for i in range(20000):
            batch=mnist.train.next_batch(50) #an iterator provided by tf
            if i %100==0:
                train_accuracy=accuracy.eval(feed_dict={
                    x:batch[0],y_:batch[1],keep_prob:1.0
                })
                print('step %d, training accuracy %g, training time %.2f secs'%(i,train_accuracy,t))
            t0=time.time()
            train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})
            t1=time.time()
            t=t1-t0

        print('test accuracy %g'%accuracy.eval(feed_dict={
            x:mnist.test.images,y_:mnist.test.labels, keep_prob:1.0
        }))
    
    
    
    
    
    
    