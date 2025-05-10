from re import M
import numpy as np


class CrossEntropyLoss():
    """Cross entropy loss function."""

    def forward(self, Y, Y_hat):
        """Computes the loss for predictions `Y_hat` given one-hot encoded labels
        `Y`.

        Parameters
        ----------
        Y      one-hot encoded labels of shape (batch_size, num_classes)
        Y_hat  model predictions in range (0, 1) of shape (batch_size, num_classes)

        Returns
        -------
        a single float representing the loss
        """
        ### YOUR CODE HERE ###
        m = Y.shape[0]  #defining the number of samples as m    (batch size)  
        L = -np.sum(Y* np.log(Y_hat))/m      # computing loss 
        
        return L  # returning loss as single float

    def backward(self, Y, Y_hat):
        """Backward pass of cross-entropy loss.
        NOTE: This is correct ONLY when the loss function is SoftMax.

        Parameters
        ----------
        Y      one-hot encoded labels of shape (batch_size, num_classes)
        Y_hat  model predictions in range (0, 1) of shape (batch_size, num_classes)

        Returns
        -------
        the gradient of the cross-entropy loss with respect to the vector of
        predictions, `Y_hat`
        """
        ### YOUR CODE HERE ###
        m= Y.shape[0]  #defining the number of samples as m  (batch size)
        dL_over_dy_hat = - (Y/Y_hat) / m     # computing the gradient of cross-entropy loss
        

        return dL_over_dy_hat # returning loss gradinet



