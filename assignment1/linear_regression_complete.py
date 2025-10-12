import numpy as np 

class LinearRegression:

    def fit(self, X, y, fit_intercept=True):
        self.fit_intercept = fit_intercept
        # add intercept
        if fit_intercept:
            ones = np.ones(len(X)).reshape(len(X), 1)
            X = np.concatenate((ones, X), axis=1)
        # Add data and dimension
        self.X = np.array(X)
        self.y = np.array(y)
        self.N, self.D = self.X.shape
        # estimate parameters with normal equations (avoid SVD which can allocate large temporaries
        # for very tall matrices). This is faster and uses much less peak memory when N >> D.
        XTX = self.X.T @ self.X
        XTy = self.X.T @ self.y
        self.beta_hats = np.linalg.solve(XTX, XTy)
        # make in-sample predictions
        self.y_hat = np.dot(self.X, self.beta_hats)
        # calculate loss
        self.L = 1/self.N*np.sum((self.y - self.y_hat)**2)

    # Prediction function
    def predict(self, X_test, predict_intercept=None):
        X_test = np.array(X_test)
        # Default to same intercept behavior as fit
        if predict_intercept is None:
            predict_intercept = self.fit_intercept
        if predict_intercept:
            ones = np.ones(len(X_test)).reshape(len(X_test), 1)
            X_test = np.concatenate((ones, X_test), axis=1)
        return np.dot(X_test, self.beta_hats)

    # Get parameteres function
    def get_params(self):
        """Return coefficients in a structured way"""
        if self.fit_intercept:
            return {
                'intercept': self.beta_hats[0],
                'coefficients': self.beta_hats[1:]
            }
        else:
            return {
                'intercept': 0.0,
                'coefficients': self.beta_hats
            }
