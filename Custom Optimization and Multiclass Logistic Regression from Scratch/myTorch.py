import numpy as np
import random
import collections
import util

# YOU ARE NOT ALLOWED TO USE sklearn or Pytorch in this assignment


class Optimizer:

    def __init__(
        self, name, lr=0.001, gama=0.9, beta_m=0.9, beta_v=0.999, epsilon=1e-8
    ):
        # self.lr will be set as the learning rate that we use upon creating the object, i.e., lr
        # e.g., by creating an object with Optimizer("sgd", lr=0.0001), the self.lr will be set as 0.0001
        self.lr = lr

        # Based on the name used for creating an Optimizer object,
        # we set the self.optimize to be the desiarable method.
        if name == "sgd":
            self.optimize = self.sgd
        elif name == "heavyball_momentum":
            # setting the gamma parameter and initializing the momentum
            self.gama = gama
            self.v = 0
            self.optimize = self.heavyball_momentum
        elif name == "nestrov_momentum":
            # setting the gamma parameter and initializing the momentum
            self.gama = gama
            self.v = 0
            self.optimize = self.nestrov_momentum
        elif name == "adam":
            # setting beta_m, beta_v, and epsilon
            # (read the handout to see what these parametrs are)
            self.beta_m = beta_m
            self.beta_v = beta_v
            self.epsilon = epsilon

            # setting the initial first momentum of the gradient
            # (read the handout for more info)
            self.v = 0

            # setting the initial second momentum of the gradient
            # (read the handout for more info)
            self.m = 0

            # initializing the iteration number
            self.t = 1

            self.optimize = self.adam

    def sgd(self, gradient):
      # TODO: add your implementation here
      v = -gradient         
      update= (self.lr) *v  # compute update vector using gradient descent algorithm
      return update      # return the update vector 
    "*** YOUR CODE ENDS HERE ***"

    def heavyball_momentum(self, gradient):
        # TODO: add your implementation here
        "*** YOUR CODE STARTS HERE ***"
        self.v = -(self.lr) *gradient + (self.gama) *(self.v)   #compute the update vector used by gradient descent algorithm
        # with heavy ball momentum 
        return self.v   # return the update vector 
        "*** YOUR CODE ENDS HERE ***"

    def nestrov_momentum(self, gradient):
        return self.heavyball_momentum(gradient)

    def adam(self, gradient):
        # TODO: add your implementation here
        "*** YOUR CODE STARTS HERE ***"
        self.m = self.beta_m * self.m + (1 - self.beta_m) * gradient   # estimate first moment
        self.v = self.beta_v * self.v + (1 - self.beta_v) * (gradient ** 2) # estimate second moment
        m_hat = self.m / (1 - self.beta_m ** self.t) # compute bias-corrected estimates
        v_hat = self.v / (1 - self.beta_v ** self.t) # compute bias-corrected estimates
        self.t += 1    # increment step time
        update = - (self.lr * m_hat) / (np.sqrt(v_hat) + self.epsilon)     # compute the update vector used by adam
        return update # return the update vector 
        "*** YOUR CODE ENDS HERE ***"


class MultiClassLogisticRegression:
    def __init__(self, n_iter=10000, thres=1e-3):
        self.n_iter = n_iter
        self.thres = thres

    def fit(
        self,
        X,
        y,
        batch_size=64,
        lr=0.001,
        gama=0.9,
        beta_m=0.9,
        beta_v=0.999,
        epsilon=1e-8,
        rand_seed=4,
        verbose=False,
        optimizer="sgd",
    ):
        # setting the random state for consistency.
        np.random.seed(rand_seed)

        # find all classes in the train dataset.
        self.classes = self.unique_classes_(y)

        # assigning an integer value to each class, from 0 to (len(self.classes)-1)
        self.class_labels = self.class_labels_(self.classes)

        # one-hot-encode the labels.
        self.y_one_hot_encoded = self.one_hot(y)

        # add a column of 1 to the leftmost column.
        X = self.add_bias(X)

        # initialize the E_in list to keep track of E_in after each iteration.
        self.loss = []

        # initialize the weight parameters with a matrix of all zeros.
        # each row of self.weights contains the weights for one of the classes.
        self.weights = np.zeros(shape=(len(self.classes), X.shape[1]))

        # create an instance of optimizer
        opt = Optimizer(
            optimizer, lr=lr, gama=gama, beta_m=beta_m, beta_v=beta_v, epsilon=epsilon
        )

        i, update = 0, 0
        while i < self.n_iter:
            self.loss.append(
                self.cross_entropy(self.y_one_hot_encoded, self.predict_with_X_aug_(X))
            )
            "*** YOUR CODE STARTS HERE ***"
            # TODO: sample a batch of data, X_batch and y_batch, with batch_size number of datapoint uniformly at random
            idx = np.random.choice(X.shape[0], batch_size, replace=False) # randomly choose indices from dataset
            X_batch, y_batch = X[idx], self.y_one_hot_encoded[idx] # get a batch of inputs and labels

            # TODO: find the gradient that should be inputed the optimization function.
            gradient = self.compute_grad(X_batch, y_batch, self.weights)


            # NOTE: for nestrov_momentum, the gradient is derived at a point different from self.weights
            # See the assignments handout or the lecture note for more information.
             
            # TODO: find the update vector by using the optimization method and update self.weights, accordingly.
            update = opt.optimize(gradient)
            self.weights+=update #update self.weights
            # TODO: stopping criterion. check if norm infinity of the update vector is smaller than self.thres.
            if np.max(np.abs(update)) < self.thres:
            # if so, break the while loop.
              break
            "*** YOUR CODE ENDS HERE ***"
            if i % 1000 == 0 and verbose:
                print(
                    " Training Accuray at {} iterations is {}".format(
                        i, self.evaluate_(X, self.y_one_hot_encoded)
                    )
                )
            i += 1
        return self

    def add_bias(self, X):
        # TODO: add your implementation here
        "*** YOUR CODE STARTS HERE ***"
        return np.insert(X, 0, 1, axis=1)  # add column of 1s to X and return the result
        
        "*** YOUR CODE ENDS HERE ***"

    def unique_classes_(self, y):
        # TODO: add your implementation here
        "*** YOUR CODE STARTS HERE ***"
        return np.unique(y)       #return a list of unique elements in y
        "*** YOUR CODE ENDS HERE ***"

    def class_labels_(self, classes):
        # TODO: add your implementation here
        "*** YOUR CODE STARTS HERE ***"
        dictionary = {} # initialize an empty dictionary 
        for value, key in enumerate(self.unique_classes_(classes)): # loop through list of 
        # unique elements in y
          dictionary[key] = value #return a dictionary with list of unique elements in 
          # y elements of the list classes as its keys and a unique integer from 0 to the 
          # total number of classed as their values
        return dictionary 

        "*** YOUR CODE ENDS HERE ***"

    def one_hot(self, y):
        # TODO: add your implementation here
        "*** YOUR CODE STARTS HERE ***"
        one_hot_encoded = np.zeros((len(y), len(self.class_labels_(y)))) # initialize one_hot matrix as zero matrix
        class_labels = self.class_labels_(y) # get unique integers and keys 
        for key, value in enumerate(y):
            one_hot_encoded[key, class_labels[value]] = 1 # set teh right index =1 in each class of y
        return one_hot_encoded     # return one hot encoded version of y
        "*** YOUR CODE ENDS HERE ***"

    def softmax(self, z):
        # TODO: add your implementation here
        "*** YOUR CODE STARTS HERE ***"
        p = np.exp(z - np.max(z, axis=1, keepdims=True))  # coverting the row of the input matrix into probability distribution
        # subtracting from max z to have a stable exp value
        return p / p.sum(axis=1, keepdims=True) # return probability distribution
        "*** YOUR CODE ENDS HERE ***"

    def predict_with_X_aug_(self, X_aug):
        # TODO: add your implementation here
        "*** YOUR CODE STARTS HERE ***"
        z = np.dot(X_aug, (np.transpose(self.weights))) # compute input matrix z 
        # for each datapoint in X based on the model’s weight parameter
        return self.softmax(z) # return compute predicted probability using input z
        "*** YOUR CODE ENDS HERE ***"

    def predict(self, X):
        # TODO: add your implementation here
        "*** YOUR CODE STARTS HERE ***"
        # compute and return predicted probability for each datapoint in X 
        #based on the model’s weight parameter when X is not augmented
        return self.predict_with_X_aug_(self.add_bias(X))
        "*** YOUR CODE ENDS HERE ***"

    def predict_classes(self, X):
        # TODO: add your implementation here
        "*** YOUR CODE STARTS HERE ***"
        y_hat = self.predict(X) # compute predicted class for each data point
        # return an array with M elements that are predicted class for each data point
        return np.array(self.classes[np.argmax(y_hat, axis=1)])
        "*** YOUR CODE ENDS HERE ***"

    def score(self, X, y):# generating prediction 
        # TODO: add your implementation here
        "*** YOUR CODE STARTS HERE ***"
        count = 0 # intialize count 
        for i in range(len(X)): # loop through datapoints in X
          if self.predict_classes(X)[i] == y[i]: # check if data point in X is correctly classfied 
            count+=1 # increment count if above is true 
        return count/len(X) # return ratio of correctly classfied datapoints
        "*** YOUR CODE ENDS HERE ***"

    def evaluate_(self, X_aug, y_one_hot_encoded): # how well a model performe on a dataset
        # TODO: add your implementation here
        "*** YOUR CODE STARTS HERE ***"
        count = 0 # intialize count 
        y_hat = np.argmax(self.predict_with_X_aug_(X_aug), axis=1) # find predicted class
        y = np.argmax(y_one_hot_encoded, axis=1)
        for i in range(len(X_aug)): # loop through datapoints in X_aug
          if y_hat[i] == y[i]: # check if data point in X is correctly classfied 
            count+=1 # increment count if above is true
        return count/len(X_aug) # return ratio of correctly classfied datapoints
        "*** YOUR CODE ENDS HERE ***"

    def cross_entropy(self, y_one_hot_encoded, probs):
        # TODO: add your implementation here
        "*** YOUR CODE STARTS HERE ***"
        loss = 0
        for i in range(len(probs)): # loop through probs
          loss += np.sum(-y_one_hot_encoded[i] * np.log(probs[i]))  # compute loss that  is differennce between predicted probabliity and actual label 
        return loss/len(probs) # return corss entropy error
        "*** YOUR CODE ENDS HERE ***"

    def compute_grad(self, X_aug, y_one_hot_encoded, w):
        # TODO: add your implementation here
        "*** YOUR CODE STARTS HERE ***"
        y_hat = self.predict_with_X_aug_(X_aug)  # find predicted probability
        error = y_hat - y_one_hot_encoded  #compute error (difference between predicted and actual labels)
        E_in_gradient = np.dot(np.transpose(error), X_aug) / len(X_aug)  # compute gradient of Ein at w
        return E_in_gradient # return E_in_gradient
        
        "*** YOUR CODE ENDS HERE ***"


def kmeans(examples, K, maxIters):
    """
    Perform K-means clustering on |examples|, where each example is a sparse feature vector.

    examples: list of examples, each example is a string-to-float dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxIters: maximum number of epochs to run (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments (i.e. if examples[i] belongs to centers[j], then assignments[i] = j),
            final reconstruction loss)
    """

    # TODO: add your implementation here
    "*** YOUR CODE STARTS HERE ***"
    pass
    "*** YOUR CODE ENDS HERE ***"
