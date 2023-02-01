import numpy as np


class LinearRegression:
    """
    Linear Regression
    """

    w: np.ndarray
    b: float

    def __init__(self):
        # raise NotImplementedError()
        self.w = None
        self.b = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        fit
        """

        # Append a column with `1` to X for bias
        X = np.hstack(np.ones((X.shape[0], 1)), X)

        # Analytical solution by Matrix Inversion if Inverse exists
        # if np.linalg.det(X.T @ X) != 0:
        sol = np.linalg.inv(X.T @ X) @ X.T @ y

        # Extracting weights and bias
        self.w = sol[1:]
        self.b = sol[0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        predict
        """
        # raise NotImplementedError()

        # Append a column with `1` to X for bias
        # X = np.hstack(np.ones((X.shape[0], 1)), X)

        # Calculate y_hat (predictions) using the weights and bias and return the predictions
        # y_hat = X @ np.hstack(self.w, self.b)

        # return X @ np.hstack(self.w, self.b)
        return X @ self.w + self.b


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None:
        """
        fit
        """

        # Extracting the size of dataset
        # n = X.shape[0]

        # Initialuze the weights
        self.w = np.zeros(X.shape[1])
        self.b = 0

        #  Gradient descent training loop
        for epoch in range(epochs):
            y_pred = X @ self.w + self.b  # Predicting the labels for X
            # dl_dw = (2 / n) * X.T @ (y_pred - y)
            # dl_db = (2 / n) * (y_pred - y)

            # self.w -= lr * dl_dw  # Update weights
            # self.b -= lr * dl_db  # Update bias
            self.w -= lr * (2 / X.shape[0]) * X.T @ (y_pred - y)  # Update weights
            self.b -= lr * (2 / X.shape[0]) * np.sum(y_pred - y)  # Update bias

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        # raise NotImplementedError()

        # Calculate predictions using the weights and bias and return the predictions
        # y_pred = X @ self.w + self.b

        return X @ self.w + self.b
