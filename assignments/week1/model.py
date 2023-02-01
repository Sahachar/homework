import numpy as np


class LinearRegression:
    '''
    Linear Regression
    '''

    w: np.ndarray
    b: float

    def __init__(self):
        # raise NotImplementedError()
        self.w = None
        self.b = None


    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        '''
        fit
        '''
        # raise NotImplementedError()
        
        # Append a column with `1` to X for bias
        X = np.hstack(X, np.ones((X.shape[0], 1)))

        # Analytical solution by Matrix Inversion if Inverse exists
        if np.linalg.det(X.T@X) != 0:
            sol = np.linalg.inv(X.T@X)@X.T@y

        # Extracting weights and bias
        self.w = sol[:-1]
        self.b = sol[-1]



    def predict(self, X: np.ndarray)->np.ndarray:
        '''
        predict
        '''
        # raise NotImplementedError()

        # Calculate y_hat (predictions) using the weights and bias
        y_hat = X@self.w + self.b

        return y_hat


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None:
        '''
        fit
        '''
        # raise NotImplementedError()

        # Extracting the size of dataset
        n = X.shape[0]  

        # Initialuze the weights
        self.w = np.zeros(X.shape[1])
        self.b = np.zeros(X.shape[0])
        
        #  Gradient descent training loop
        for epoch in range(epochs):
            y_pred = X@self.w + self.b    # Predicting the labels for X
            dl_dw = (2/n)*X.T@(y_pred-y)
            dl_db = (2/n)*(y_pred-y)
            self.w -= lr*dl_dw  # Update weights
            self.b -= lr*dl_db  # Update bias 



    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        # raise NotImplementedError()

        # # Calculate predictions using the weights and bias
        y_pred = X@self.w + self.b

        return y_pred 
