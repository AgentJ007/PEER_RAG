"""
Machine learning models module.
"""
import numpy as np

class OrdinalLogisticRegression:
    """
    Dummy Ordinal Logistic Regression model.
    """
    def predict_proba(self, X):
        # Return dummy probabilities for 3 classes: Accept, Minor Revision, Major Revision
        n_samples = X.shape[0] if hasattr(X, 'shape') else 1
        return np.array([[0.1, 0.3, 0.6]] * n_samples)

class KNearestNeighbors:
    """
    Dummy K-Nearest Neighbors model.
    """
    def predict_proba(self, X):
        # Return dummy probabilities
        n_samples = X.shape[0] if hasattr(X, 'shape') else 1
        return np.array([[0.2, 0.4, 0.4]] * n_samples)
