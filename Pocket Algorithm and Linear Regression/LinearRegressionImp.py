import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def fit_LinRegr(X_train, y_train):
    """
    This function computes the parameters w of a linear plane which best fits
    the training dataset.

    Parameters
    ----------
    X_train: numpy.ndarray with shape (N,d)
        represents the matrix of input features where N is the total number of 
        training samples and d is the dimension of each input feature vector.
    y_train: numpy.ndarray with shape (N,)
        the ith component represents the output observed in the training set
        for the ith row in X_train matrix which corresponds to the ith input
        feature. Each element in y_train takes a Real value.
    
    Returns
    -------
    w: numpy.ndarray with shape (d+1,)
        represents the coefficients of the line computed by the pocket
        algorithm that best separates the two classes of training data points.
        The dimensions of this vector is (d+1) as the offset term is accounted
        in the computation.
    """

    # TODO: add your implementation here
    "*** YOUR CODE HERE ***"
    row, col = X_train.shape #To identify the number of rows and columns in a matrix
    ones_column = np.ones((row, 1))  # create a colum vector of ones with row# of rows. 
    aug_x = np.hstack([ones_column, X_train]) #To stack two arrays (stacking ones_column to the left side of X_train)
    X_transpose = np.transpose(aug_x)  # Find X transpose 
    X_t = np.dot(X_transpose, aug_x) # Calculate X_transpose * aug_x
    X_t_inverse = np.linalg.pinv(X_t) # Find the pseudo-inverse of X_t
    w = np.dot((np.dot(X_t_inverse, X_transpose)), y_train)  # compute w finally as (X^TX)^-1 X^Ty

    return w # return computed wieght vector 

def mse(X_train, y_train, w):
    """
    This function finds the mean squared error introduced by the linear plane
    defined by w.

    Parameters
    ----------
    X_train: numpy.ndarray with shape (N,d)
        represents the matrix of input features where N is the total number of 
        training samples and d is the dimension of each input feature vector.
    y_train: numpy.ndarray with shape (N,)
        the ith component represents the output observed in the training set
        for the ith row in X_train matrix which corresponds to the ith input
        feature. Each element in y_train takes a Real value.
    w: numpy.ndarray with shape (d+1,)
        represents the coefficients of a linear plane.

    Returns
    -------
    avgError: float
        he mean squared error introduced by the linear plane defined by w.
    """

    # TODO: add your implementation here
    # HINT: You may find the implemented pred function useful. The pred function 
    # finds the prediction by the linear regression model defined by w for the
    # input datapoint x_i.
    "*** YOUR CODE HERE ***"
    N = X_train.shape[0] # Identify number of rows in X_train matrix 
    row, col = X_train.shape
    ones_column = np.ones((N, 1))  # create a colum vector of ones with row# of rows. 
   
    aug_x = np.hstack([ ones_column, X_train]) #To stack two arrays (stacking ones_column to the left side of X_train)
    error =0 # Initialize error as 0
    for i in range(N): # looping through all rows
        y_predicted = pred(aug_x[i], w) # finding predicted value using pred function 
        error += (y_train[i] - y_predicted)**2 #calculate the squared error for the i-th data point and then adding it to the cumulative error
    avgError = error/N #finding the avg error by dividing the acculmated error by number of rows
    return avgError # return mean squared error introduced by the linear plane defined by w over dataset


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

    pred_i = np.dot(x_i, w)  # finding the prediction by the classifier defined by w.
    return pred_i


def test_SciKit(X_train, X_test, y_train, y_test):
    """
    This function will output the mean squared error on the test set, which is
    obtained from the mean_squared_error function imported from sklearn.metrics
    library to report the performance of the model fitted using the 
    LinearRegression model available in the sklearn.linear_model library.
    
    Parameters
    ----------
    X_train: numpy.ndarray with shape (N,d)
        Represents the matrix of input features where N is the total number of 
        training samples and d is the dimension of each input feature vector.
    X_test: numpy.ndarray with shape (M,d)
        Represents the matrix of input features where M is the total number of
        testing samples and d is the dimension of each input feature vector.
    y_train: numpy.ndarray with shape (N,)
        The ith component represents output observed in the training set for the
        ith row in X_train matrix which corresponds to the ith input feature.
    y_test: numpy.ndarray with shape (M,)
        The ith component represents output observed in the test set for the
        ith row in X_test matrix which corresponds to the ith input feature.
    
    Returns
    -------
    error: float
        The mean squared error on the test set.
    """

    # TODO: initiate an object of the LinearRegression type. 
    "*** YOUR CODE HERE ***"
    object = LinearRegression() #create an object of the LinearRegression type

    # TODO: run the fit function to train the model. 
    "*** YOUR CODE HERE ***" 
    object.fit(X_train, y_train) #train linear regression model on dataset

    # TODO: use the predict function to perform predictions using the trained
    # model. 
    "*** YOUR CODE HERE ***"
    y_predict = object.predict(X_test) #predict outputs for x_test using the model

    # TODO: use the mean_squared_error function to find the mean squared error
    # on the test set. Don't forget to return the mean squared error.
    "*** YOUR CODE HERE ***"
    error = mean_squared_error(y_test, y_predict) #find the mean squared error (MSE) which is the average of the squared differences between the predicted and true values

    return error # return the mean squared error on the test set


def subtestFn():
    """
    This function tests if your solution is robust against singular matrix.
    X_train has two perfectly correlated features.
    """

    X_train = np.asarray([[1, 2],
                          [2, 4],
                          [3, 6],
                          [4, 8]])
    y_train = np.asarray([1, 2, 3, 4])
    
    try:
      w = fit_LinRegr(X_train, y_train)
      print("weights: ", w)
      print("NO ERROR")
    except:
      print("ERROR")


def testFn_Part2():
    """
    This function loads diabetes dataset and splits it into train and test set.
    Then it finds and prints the mean squared error from your linear regression
    model and the one from the scikit library.
    """

    X_train, y_train = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)
    
    w = fit_LinRegr(X_train, y_train)
    
    #Testing Part 2a
    e = mse(X_test, y_test, w)
    
    #Testing Part 2b
    scikit = test_SciKit(X_train, X_test, y_train, y_test)
    
    print(f"Mean squared error from Part 2a is {e}")
    print(f"Mean squared error from Part 2b is {scikit}")


if __name__ == "__main__":
    print (f"{12*'-'}subtestFn{12*'-'}")
    subtestFn()

    print (f"{12*'-'}testFn_Part2{12*'-'}")
    testFn_Part2()
