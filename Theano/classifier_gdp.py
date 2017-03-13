import theano
import numpy
import sys
import collections

floatX = theano.config.floatX

#### class to build the classifier
class classifier(object):
    def __init__(self,n_features):
        random_seed=12
        num_hidden_layer=5
        l2_regularize=0.001

        #random generation
        rand= numpy.random.RandomState(random_seed)

        ###variables for the network
        input_vec= theano.tensor.fvector('input_vec')
        target= theano.tensor.fscalar('target')
        learningrate= theano.tensor.fscalar('learningrate')

        ## input-hidden layer
        w_hidden= theano.shared(numpy.asarray(rand.normal(loc=0.0,scale=0.1,size=(n_features,num_hidden_layer)),dtype=floatX),'w_hidden')

        ### Hidden layer calculation
        l1= theano.tensor.dot(input_vec,w_hidden)
        l1= theano.tensor.nnet.sigmoid(l1)

        ## hidden- output layer
        w_output = theano.shared(numpy.asarray(rand.normal(loc=0.0, scale=0.1, size=(num_hidden_layer,1)), dtype=floatX),'w_output')

        ## prediction value
        pred= theano.tensor.dot(l1,w_output)
        pred = theano.tensor.nnet.sigmoid(pred)
        print(pred)

        ### cost function

        cost= theano.tensor.sqr(pred-target).sum()
        cost+= l2_regularize*(theano.tensor.sqr(w_hidden).sum()+theano.tensor.sqr(w_output).sum())
        print (cost)

        #### gradient evaluation

        param=[w_hidden,w_output]
        gradient= theano.tensor.grad(cost,param)
        update=[(x,x-(learningrate*z))for x,z in zip(param,gradient)]

        ### define Theano function
        self.train= theano.function([input_vec,target,learningrate ],[cost,pred],updates=update,allow_input_downcast=True)
        self.test= theano.function([input_vec,target],[cost,pred],allow_input_downcast=True)

def read_data(path):
    data=[]
    with open(path,'r') as f:
        for line in f:
            line_split= line.strip().split()
            label= float(line_split[0])
            features= numpy.array([float(line_split[i]) for i in range (1,len(line_split))])
            data.append((label,features))
    return data

if __name__=="__main__":
    ## training parameters
    learningrate=0.1
    epochs=20

    ## read files

    data_train= read_data("gdp-normalized-train.txt")
    data_test= read_data("gdp-normalized-test.txt")

    ## network creation
    n_features= len(data_train[0][1])
    classifier= classifier(n_features)

    ##training
    for epoch in range(epochs):
        cost_final=0.0
        correct=0
        for label,features in data_train:
            cost, pred_val = classifier.train(features,label,learningrate)
            cost_final += cost
            if (label ==1.0 and pred_val>0.5 ) or (label ==0.0 and pred_val<0.5 ):
                correct +=1

        print("epoch"+str(epoch)+"train_cost:"+str(cost_final)+"train_accuracy:"+str(float(correct)/len(data_train)))

    ### testing the model
    cost_sum = 0.0
    correct = 0
    for label, vector in data_test:
        cost, predicted_value = classifier.test(vector, label)
        cost_sum += cost
        if (label == 1.0 and predicted_value >= 0.5) or (label == 0.0 and predicted_value < 0.5):
            correct += 1
    print("Test_cost: " + str(cost_sum) + ", Test_accuracy: " + str(float(correct) / len(data_test)))
