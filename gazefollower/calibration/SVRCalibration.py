# encoding=utf-8
# Author: GC Zhu
# Email: zhugc2016@gmail.com

import pathlib
from typing import Tuple, Any

import cv2 as cv
import numpy as np
from numpy import ndarray

from .Calibration import Calibration
from ..logger import Log


class SVRCalibration(Calibration):
    def __init__(self, model_save_path: str = ""):
        """
        Initializes the Calibration class with two SVM models for x and y coordinates.
        :param model_save_path: default path is {HOME}/GazeFollower/Calibration when `model_save_path` == ""
        """
        super().__init__()

        if model_save_path == "":
            self.workplace_calibration_dir = pathlib.Path.home().joinpath("GazeFollower", "calibration")
            if not self.workplace_calibration_dir.exists():
                self.workplace_calibration_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.workplace_calibration_dir = pathlib.Path(model_save_path)

        self.svr_x_path = self.workplace_calibration_dir.joinpath("svr_x.xml")
        self.svr_y_path = self.workplace_calibration_dir.joinpath("svr_y.xml")

        self.svr_x = cv.ml.SVM.create()  # SVM model for the x coordinate
        self.svr_y = cv.ml.SVM.create()  # SVM model for the y coordinate

        if self.svr_y_path.exists() and self.svr_x_path.exists():
            self.svr_x.load(str(self.svr_x_path))
            self.svr_y.load(str(self.svr_y_path))
            self.has_calibrated = True
        else:
            # Set default parameters for the SVM models
            self._set_svm_params(self.svr_x)
            self._set_svm_params(self.svr_y)
            self.has_calibrated = False

    @staticmethod
    def _set_svm_params(svr):
        """
        Sets the default SVM parameters for the model.
        :param svr: The SVM model to set parameters for.
        """
        svr.setType(cv.ml.SVM_EPS_SVR)
        svr.setKernel(cv.ml.SVM_RBF)
        svr.setC(1.0)  # Example C value, adjust as needed
        svr.setGamma(0.005)  # Example gamma value, adjust as needed
        svr.setP(0.001)  # Epsilon for loss function
        term_criteria = (cv.TERM_CRITERIA_MAX_ITER, 10000, 1e-4)
        svr.setTermCriteria(term_criteria)

    def predict(self, features, estimated_coordinate) -> Tuple:
        """
        Predicts the x and y coordinates using the trained SVM models.

        :param features: np.array of shape (1, m), where m is the number of features.
                        This represents a single sample feature vector for prediction.
        :param estimated_coordinate: the estimated coordinate from the gaze estimation model.
        :return: A tuple containing the predicted x and y coordinates calibrated, [x_pred, y_pred].
        """
        # Ensure the feature is a numpy array with the correct dtype and shape
        features = np.array(features, dtype=np.float32).reshape(1, -1)
        features = features[:, :]

        # Check if models are trained before making predictions
        if not self.has_calibrated:
            Log.d("SVM models are not trained. It will return estimated coordinate from gaze estimation model.")
            return self.has_calibrated, estimated_coordinate

        # Predict x and y coordinates using the trained SVM models
        predicted_x = self.svr_x.predict(features)[1].flatten()[0]
        predicted_y = self.svr_y.predict(features)[1].flatten()[0]

        # Return the predictions as a list
        return self.has_calibrated, (predicted_x, predicted_y)

    def calibrate(self, features, labels, ids=None):
        # Ensure that features and labels are numpy arrays
        features = features.astype(np.float32)
        labels = labels.astype(np.float32)

        # Split labels into x and y components
        labels_x = labels[:, 0].reshape(-1, 1)  # Extract x labels
        labels_y = labels[:, 1].reshape(-1, 1)  # Extract y labels

        try:
            # Train SVM models
            self.svr_x.train(features, cv.ml.ROW_SAMPLE, labels_x)
            self.svr_y.train(features, cv.ml.ROW_SAMPLE, labels_y)
            self.has_calibrated = True
        except Exception as e:
            self.has_calibrated = False

            Log.e("Failed to train SVM model: {}".format(e.args))
            Log.d("Try to delete previously trained model.")
            if self.svr_x_path.exists():
                self.svr_x_path.unlink()  # Deletes svr_x.xml
                Log.d(f"Deleted: {self.svr_x_path}")
            else:
                Log.d(f"No trained model found at {self.svr_x_path}")

            if self.svr_y_path.exists():
                self.svr_y_path.unlink()  # Deletes svr_y.xml
                Log.d(f"Deleted: {self.svr_y_path}")
            else:
                Log.d(f"No trained model found at {self.svr_y_path}")

        if self.has_calibrated:
            # Predict using the trained models
            predicted_x = self.svr_x.predict(features)[1]
            predicted_y = self.svr_y.predict(features)[1]
            # Calculate the Euclidean distance between the predicted and actual labels
            euclidean_distances = np.sqrt((labels_x - predicted_x) ** 2 + (labels_y - predicted_y) ** 2)
            # Calculate the mean Euclidean error
            mean_euclidean_error = np.mean(euclidean_distances)
            Log.d(f"Calibration completed with mean Euclidean error: {mean_euclidean_error:.4f}")
            predicted_x.reshape(-1, 1)
            predicted_y.reshape(-1, 1)
            predictions = np.concatenate((predicted_x, predicted_y), axis=1)
            return self.has_calibrated, mean_euclidean_error, predictions
        else:
            return self.has_calibrated, float('inf'), None,

    def save_model(self) -> bool:
        """
        Saves the trained SVM models to XML files.

        :return: True if the models were saved, False otherwise.
        """

        # Check if the SVM models are trained
        if self.svr_x.isTrained() and self.svr_y.isTrained():
            self.svr_x.save(str(self.svr_x_path))
            self.svr_y.save(str(self.svr_y_path))
            Log.d(f"SVR model for x coordinate saved at: {self.svr_x_path}")
            Log.d(f"SVR model for y coordinate saved at: {self.svr_y_path}")
            return True
        else:
            Log.d("SVR model for x coordinate has not been trained yet.")
            Log.d("SVR model for y coordinate has not been trained yet.")
            return False

    def release(self):
        pass
