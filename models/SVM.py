import numpy as np
import scipy
from tqdm import tqdm

class SVM():
    def __init__(self, x):
        """
        Initialises SVM classifier with initializing 
        weights, alpha(learning rate), number of epochs
        and regularization constant.
        """
        self.W = np.zeros((x+1,10))   
        self.alpha = 0.00000000001
        self.epochs = 200
        self.reg_const = 0.01
        
    def calc_gradient(self, W, X, y, reg):
        
        """
        Calculate gradient of the SVM loss
          
        Inputs have dimension D, there are C classes, and we operate on minibatches
        of N examples.

        Inputs:
        - X_train: A numpy array of shape (N, D) containing a minibatch of data.
        - y_train: A numpy array of shape (N,) containing training labels; y[i] = c means
          that X[i] has label c, where 0 <= c < C.

        Returns:
        - gradient with respect to weights W; an array of same shape as W
        """
        
        C = W.shape[1]
        N = X.shape[0]
        n= X.shape[1]
        loss = 0.0

        grad_W = np.zeros(W.shape) 
        Class_scores = np.dot(X, W)    # (N, C)
        Class_Preds = Class_scores[np.arange(N), y] # (N, )
        margins = np.maximum(Class_scores - Class_Preds.reshape(N, 1) + 1.0, 0) 
        margins[np.arange(N), y] = 0
        
        loss = np.sum(margins) / N
        loss += 0.5 * reg * np.sum(W * W)
        dscores = np.zeros_like(Class_scores)  # (N, C)
        dscores[margins > 0] = 1  #classes wich needs the seconed update
        dscores[np.arange(N), y] -= np.sum(dscores, axis=1)   #  (N, 1) = (N, 1)
        
        grad_W = np.dot(X.T, dscores)
        grad_W /= N
        grad_W += reg* W 
        
        return [grad_W, loss]
    
    
    def get_acc(self, pred, y_test):
        return np.sum(y_test==pred)/len(y_test)*100 
    
    
    def train(self, X_train, y_train, X_test,y_test):
        """
        Train SVM classifier using stochastic gradient descent.

        Inputs:
        - X_train: A numpy array of shape (N, D) containing training data;
        N examples with D dimensions
        - y_train: A numpy array of shape (N,) containing training labels;
        
        Hint : Operate with Minibatches of the data for SGD
        
        """
        K = np.arange(self.epochs)
        H = np.arange(self.epochs)

        F = y_train.shape[0]
        for s in tqdm(range(self.epochs)):
            for i in range(0, F, 10):
                grad_W = self.calc_gradient(self.W, np.insert(X_train[i:i+10], 0, 1, axis=1), y_train[i:i+10], self.reg_const)
                self.W = self.W  - self.alpha * grad_W[0]
            K[s] = self.get_acc(self.predict(X_test),y_test)
            H[s] = grad_W[1]
        return [K,H]

    def predict(self, X_test):
        """
        Use the trained weights of svm classifier to predict labels for
        data points.

        Inputs:
        - X_test: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.

        Returns:
        - pred: Predicted labels for the data in X_test. pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.
        """
        X_test = np.insert(X_test, 0, 1, axis=1)
        #print(X_test.shape)
        Y_out = self.W.T.dot(X_test.T)
        pred = np.argmax(Y_out, axis=0)
        
        return pred