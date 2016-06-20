# nnet-python

This is a Python implementation of a standard feed-forward neural network with variable architecture and fixed-learning-rate gradient descent to train. I did this as an exercise, the final product is rather slow. My C++ version is much faster. It illustrates the algorithm in an intuitive fashion, based on mathematical vector and matrix computations. 

###Pre-requisites

What you need are sets of predictors (list of lists) with associated labels (list), one each for training, validating, and testing. There are helper functions included in the module for reading in data with simple csv format.

A network architecture is also required, which is simply a list of integers. The first value should equal the dimension of your predictor date. The next values are discretionary to specify the number of nodes within each layer of the network. The final value should always be 1 or the model won't predict correctly. For example an architecture of d = [15,10,5,1] represents a network accepting 15-dimensional data, with a layer with 10 nodes, and a second layer of 5 nodes. 

###Training

With data sets and architecture in hand, one can build and train a neural network. To initialize the network, simply pass in the architecture to the network constructor. As in "network = chosen_namespace.nnet(d)." Once the network is initialized, pass the appropriate data sets and parameters to the training method. For example: "network.train(dataTrain,labelTrain,dataValidate,labelValidate,learningRate,maxIteration)"

This will probably take a while, depending on the number of data points and iterations. Once the training is complete, the optimal validation error will be printed. This is a regression error, not a classification error. With a trained network in hand, one can use it for prediction. Simply call the predict method on the test/other data set. For example:
"prediction = network.predict(dataTest)".

You can then use the resulting prediction to assess error as desired. 
