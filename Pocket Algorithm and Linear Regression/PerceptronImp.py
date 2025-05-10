import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_iris


def fit_perceptron(X_train, y_train, max_epochs=5000):
    """
    This function computes the parameters w of a linear plane which separates
    the input features from the training set into two classes specified by the
    training dataset.

    Parameters
    ----------
    X_train: numpy.ndarray with shape (N,d)
        represents the matrix of input features where N is the total number of 
        training samples and d is the dimension of each input feature vector.
    y_train: numpy.ndarray with shape (N,)
        the ith component represents the output observed in the training set
        for the ith row in X_train matrix which corresponds to the ith input
        feature. Each element in y_train takes the value +1 or −1 to represent
        the first class and second class respectively.
    
    Returns
    -------
    w: numpy.ndarray with shape (d+1,)
        represents the coefficients of the line computed by the pocket
        algorithm that best separates the two classes of training data points.
        The dimensions of this vector is (d+1) as the offset term is accounted
        in the computation.
    """

    row, col = X_train.shape #To identify the number of rows and columns in a matrix
    Ein_best = 1000 # Initialzie Ein_best. 

    # TODO: initialize the weight vector w
    "*** YOUR CODE HERE ***"
    weight = np.zeros(col+1) # Intialize the weight as zero vector with size of (col+1), where col is number of features in dataset and 1 is accounted for bias or threshold. 

    # TODO: Augment X_train so that it would have an additional column of 1's.
    "*** YOUR CODE HERE ***"
    ones_column = np.ones((row, 1))  # create a colum vector of ones with row# of rows. 
    new_x = np.hstack([ones_column,X_train]) #To stack two arrays (stacking ones_column to the left side of X_train)
  
    # TODO: go over the entire dataset for max_epochs number of times. 
    # In each epoch, we examin all datapoints one by one and update w if neccessary. 
    # If after update you have a new best E_in, then save the updated weight in
    # your pocket to return it at the end
    "*** YOUR CODE HERE ***"
    for i in range(max_epochs): # Loop through all max_epochs
        for j in range(row): # Loop through all rows in the matrix 
            output = np.dot(new_x[j], weight) #start calculating prediction value by taking dot product
            y_predicted = np.sign(output) #checking the sign of dot product if it is +1 or -1
            if y_predicted != y_train[j]: # check if true value is same as predicted value.
                weight = weight + y_train[j]* new_x[j] # Update the wieght vector 
                Ein = errorPer(new_x, y_train, weight) # check in for in sample error (E_in) using errorPer function. 
                if Ein < Ein_best: # comparing if the model with new updated weight's E_in is better then Ein_best or not
                    Ein_best = Ein # if it is smaller then we set the best value as E_in of updated weight
                    w = np.copy(weight) # we store the copy of the updated weight. 
                
    return w # return best wieght after training 

def errorPer(X_train, y_train, w):
    """
    This function finds the average number of points that are misclassified by
    the plane defined by w.

    Parameters
    ----------
    X_train: numpy.ndarray with shape (N,d+1)
        Represents the matrix of input features where N is the total number of 
        training samples and d is the dimension of each input feature vector.
        Note the additional dimension which is for the additional column of ones
        added to the front of the original input.
    y_train: numpy.ndarray with shape (N,)
        The ith component represents the output observed in the training set
        for the ith row in X_train matrix which corresponds to the ith input
        feature.
    w: numpy.ndarray with shape (d+1,)
        Represents the coefficients of a linear plane.
    
    Returns
    -------
    avgError: float
        The average number of points that are misclassified by the plane
        defined by w.
    """
    # TODO: add your implementation here
    "*** YOUR CODE HERE ***"
    N = X_train.shape[0] # Identify number of rows in X_train matrix 
    count = 0 # Initialize number of misclassified points as count 
    for i in range(N): # Loop through all rows in the matrix 
        y_predicted = np.sign(np.dot(w, X_train[i])) # compute predicted value of y and see if it is +1, -1 
        if y_train[i]!= y_predicted: # compariing true value with predicted value and checking if they are not equal 
            count +=1 # counting number of missclassifed points 
    avgError = count/N # finding avg value of number of missclassifed points
    return avgError # return The average number of points that are misclassified by the plane defined by w.

def pred(x_i, w):
    """
    This function finds finds the prediction by the classifier defined by w.

    Parameters
    ----------
    x_i: numpy.ndarray with shape (d+1,)
        Represents the feature vector of (d+1) dimensions of the ith test
        datapoint.
    w: numpy.ndarray with shape (d+1,)
        Represents the coefficients of a linear plane.
    
    Returns
    -------
    pred_i: int
        The predicted class.
    """
    pred_i = -1 if np.dot(x_i, w) < 0 else 1
    return pred_i
  
def confMatrix(X_train, y_train, w):
    """
    This function populates the confusion matrix. 

    Parameters
    ----------
    X_train: numpy.ndarray with shape (N,d)
        Represents the matrix of input features where N is the total number of 
        training samples and d is the dimension of each input feature vector.
    y_train: numpy.ndarray with shape (N,)
        The ith component represents the output observed in the training set
        for the ith row in X_train matrix which corresponds to the ith input
        feature.
    w: numpy.ndarray with shape (d+1,)
        Represents the coefficients of a linear plane.
    
    Returns
    -------
    conf_mat: numpy.ndarray with shape (2,2), composed of integer values
        - conf_mat[0, 0]: True Negative
            number of points correctly classified to be class −1.
        - conf_mat[0, 1]: False Positive
            number of points that are in class −1 but are classified to be class
            +1 by the classifier.
        - conf_mat[1, 0]: False Negative
            number of points that are in class +1 but are classified to be class
            −1 by the classifier.
        - conf_mat[1, 1]: True Positive
            number of points correctly classified to be class +1.
    """

    # TODO: add your implementation here
    "*** YOUR CODE HERE ***"
    row = X_train.shape[0] #Identify number of rows in X_train matrix 
    # Augment X_train so that it would have an additional column of 1's.
    ones_column = np.ones((row, 1))
    new_x = np.hstack([ones_column, X_train])

    conf_mat = np.zeros((2,2), dtype = int) #Initializing confusion matrix as zero matrix with shape (2,2), composed of integer values. 

    for i in range(row): # Looping through rows of matrix. 
        predicted_y = np.sign(np.dot(w, new_x[i]))  #calculating the predicted value of y checking if it is +1 or -1

        #Counting number of True Negative by checking if both predicted value and true value of output is -1 
        if predicted_y == -1 and y_train[i] == -1: 
            conf_mat[0, 0] += 1  

        #Counting number of False Positive by checking if true values are in class −1 but are classified to be class +1 by the classifier.
        elif predicted_y == 1 and y_train[i] == -1:
            conf_mat[0, 1] += 1  

        #Counting number of False Negative by checking if true values are in class 1 but are classified to be class -1 by the classifier.
        elif predicted_y == -1 and y_train[i] == 1:
            conf_mat[1, 0] += 1  

        #Counting number of True Positive by checking if both predicted value and true value of output is 1 
        elif predicted_y == 1 and y_train[i] == 1:
            conf_mat[1, 1] += 1  

    return conf_mat # returning confusion matrix after counting number of TN, FP, FN, TP as [[TN, FP],[FN, TP]]

def test_SciKit(X_train, X_test, y_train, y_test):    
    """
    This function uses Perceptron imported from sklearn.linear_model to fit the
    linear classifer using the Perceptron learning algorithm. Then it returns
    the result obtained from the confusion_matrix function imported from
    sklearn.metrics to report the performance of the fitted model.
    
    Parameters
    ----------
    X_train: numpy.ndarray with shape (N,d)
        Represents the matrix of input features where N is the total number of 
        training samples and d is the dimension of each input feature vector.
    X_test: numpy.ndarray with shape (M,d)
        Represents the matrix of input features where M is the total number of
        testing samples and d is the dimension of each input feature vector.
    y_train: numpy.ndarray with shape (N,)
        The ith component represents the output observed in the training set
        for the ith row in X_train matrix which corresponds to the ith input
        feature.
    y_test: numpy.ndarray with shape (M,)
        The ith component represents the output observed in the test set
        for the ith row in X_test matrix.
    
    Returns
    -------
    conf_mat: numpy.ndarray with shape (2,2), composed of integer values
        - conf_mat[0, 0]: True Negative
            number of points correctly classified to be class −1.
        - conf_mat[0, 1]: False Positive
            number of points that are in class −1 but are classified to be class
            +1 by the classifier.
        - conf_mat[1, 0]: False Negative
            number of points that are in class +1 but are classified to be class
            −1 by the classifier.
        - conf_mat[1, 1]: True Positive
            number of points correctly classified to be class +1.
    """

    # TODO: initiate an object of the Perceptron type. 
    "*** YOUR CODE HERE ***"
    clf = Perceptron(tol=1e-3, max_iter=5000) #initiate an object of the Perceptron type with deafult value of tolerance and max_iter. 
    
    # TODO: run the fit function to train the classifier. 
    "*** YOUR CODE HERE ***" 
    clf.fit(X_train, y_train) #Fit the model to training data whic is X_train and y_train

    # TODO: use the predict function to perform predictions using the trained
    # algorithm. 
    "*** YOUR CODE HERE ***"
    y_pred = clf.predict(X_test)  # predict output of test data using trained classier 

    # TODO: Use the confusion_matrix function to find the confusion matrix.
    # Don't forget to return the confusion matrix.
    "*** YOUR CODE HERE ***"

    # generate confusion matrix as below by comapring test data with predicted data. 
    conf_mat= confusion_matrix(y_test, y_pred)   # we use y_test here since in confusion matrix we want to check performance of model on untrained or test data 

    return conf_mat # return created confusion matrix that 

def test_Part1():
    """
    This is the main routine function. It loads IRIS dataset, picks its last
    100 datapoints and split them into train and test set. Then finds and prints
    the confusion matrix from part 1a and 1b.
    """

    # Loading and splitting IRIS dataset into train and test set
    X_train, y_train = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_train[50:],
                                                        y_train[50:],
                                                        test_size=0.2,
                                                        random_state=42)

    # Set the labels to +1 and -1. 
    # The original labels in the IRIS dataset are 1 and 2. We change label 2 to -1. 
    y_train[y_train != 1] = -1
    y_test[y_test != 1] = -1

    # Pocket algorithm using Numpy
    w = fit_perceptron(X_train, y_train)
    my_conf_mat = confMatrix(X_test, y_test, w)

    # Pocket algorithm using scikit-learn
    scikit_conf_mat = test_SciKit(X_train, X_test, y_train, y_test)
    
    # Print the result
    print(f"{12*'-'}Test Result{12*'-'}")
    print("Confusion Matrix from Part 1a is: \n", my_conf_mat)
    print("\nConfusion Matrix from Part 1b is: \n", scikit_conf_mat)
    

if __name__ == "__main__":
    test_Part1()