# Multi-class classification of hand-written digits od Mnist dataset

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data/',one_hot=True)

# define parameters for the number of neurons in different hidden layer , outputlayer and size of the batch
l1_hidden=500
l2_hidden=500
l3_hidden=500
n_class=10
batch_sixe=100

### defining input x and output label y
x= tf.placeholder('float',[None,784])
y= tf.placeholder('float')

# define each layer with their weights and bias
def neural_network(data):
    hidden_l1 ={"weights": tf.Variable(tf.random_normal([784,l1_hidden])),
                "bias":tf.Variable(tf.random_normal([l1_hidden]))}
    hidden_l2= {"weights": tf.Variable(tf.random_normal([l1_hidden,l2_hidden])),
                 "bias": tf.Variable(tf.random_normal([l2_hidden]))}
    hidden_l3= {"weights": tf.Variable(tf.random_normal([l2_hidden, l3_hidden])),
                 "bias": tf.Variable(tf.random_normal([l3_hidden]))}
    output_layer= {"weights": tf.Variable(tf.random_normal([l3_hidden,n_class])),
                 "bias": tf.Variable(tf.random_normal([n_class]))}

# Neural network operations
    l1= tf.add(tf.matmul(data,hidden_l1['weights']),hidden_l1['bias'])
    l1=tf.nn.relu(l1)
    l2 = tf.add(tf.matmul(l1, hidden_l2['weights']), hidden_l2['bias'])
    l2 = tf.nn.relu(l2)
    l3 = tf.add(tf.matmul(l2, hidden_l3['weights']), hidden_l3['bias'])
    l3 = tf.nn.relu(l3)
    output= tf.matmul(l3,output_layer['weights'])+output_layer['bias']
    return output

# function to call the Neural network and to evaluate the cost and optimization using Adam Optimizer
def train_model(x):
    pred= neural_network(x)
    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
    optimizer= tf.train.AdamOptimizer().minimize(cost)

# numner of iteration of running through the feed-forward and back propogation
    epochs=10

# beginning of the session to run the computational graph
    with tf.Session()as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):
            loss=0
            for _ in range(int(mnist.train.num_examples/batch_sixe)):
                ### training features and labels
                epoch_x,epoch_y= mnist.train.next_batch(batch_sixe)
                #### running the network
                _,c=sess.run([optimizer,cost],feed_dict={x:epoch_x,y:epoch_y})
                loss+=c

            print("epoch",epoch,'loss:',loss)
        # evaluating the accuracy of the model
        correct= tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
        accuracy= tf.reduce_mean(tf.cast(correct,'float'))
        # evaluating the model on the test data
        print("accuracy:",accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))

train_model(x)

