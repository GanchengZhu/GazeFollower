# encoding=utf-8
# Author: GC Zhu
# Email: zhugc2016@gmail.com

import numpy as np
from typing import Tuple
from .Calibration import Calibration
from ..logger import Log

class MultivariateRidgeCalibration(Calibration):
    def __init__(self, alpha: float = 0.1):
        """
        Initializes the Calibration class with a single Multivariate Ridge Regression model for both x and y.
        :param alpha: Regularization strength (lambda)
        """
        super().__init__()
        self.alpha = alpha
        self.weights = None
        self.has_calibrated = False

    def predict(self, features, estimated_coordinate) -> Tuple:
        """
        Predicts the x and y coordinates simultaneously.
        """
        features = np.array(features, dtype=np.float32).reshape(1, -1)
        if not self.has_calibrated:
            Log.d("Ridge model is not trained. Returning estimated coordinate.")
            return self.has_calibrated, estimated_coordinate
        
        # Add bias term (column of 1s)
        features_b = np.c_[np.ones((features.shape[0], 1)), features]
        predicted = features_b @ self.weights
        
        return self.has_calibrated, (predicted[0, 0], predicted[0, 1])

    def calibrate(self, features, labels, ids=None):
        features = features.astype(np.float32)
        labels = labels.astype(np.float32)
        
        # Add bias term (column of 1s)
        X_b = np.c_[np.ones((features.shape[0], 1)), features]
        n, d = X_b.shape
        
        # Identity matrix for regularization
        I = np.eye(d)
        I[0, 0] = 0 # Do not regularize the bias term
        
        try:
            # W = (X^T X + alpha * I)^{-1} X^T Y
            XtX = X_b.T @ X_b
            XtY = X_b.T @ labels
            self.weights = np.linalg.inv(XtX + self.alpha * I) @ XtY
            self.has_calibrated = True
        except Exception as e:
            self.has_calibrated = False
            Log.e(f"Failed to train Multivariate Ridge Regression: {e}")
            
        if self.has_calibrated:
            predictions = X_b @ self.weights
            euclidean_distances = np.sqrt(np.sum((labels - predictions) ** 2, axis=1))
            mean_euclidean_error = np.mean(euclidean_distances)
            Log.d(f"Calibration completed with mean Euclidean error: {mean_euclidean_error:.4f}")
            return self.has_calibrated, mean_euclidean_error, predictions
        else:
            return self.has_calibrated, float('inf'), None
            
    def save_model(self) -> bool:
        # Saving weights is straightforward using numpy if needed.
        return False
        
    def release(self):
        pass
