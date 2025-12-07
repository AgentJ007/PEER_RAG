import pickle
import numpy as np

class OrdinalLogisticRegression:
    def predict_proba(self, X):
        n_samples = X.shape[0] if hasattr(X, 'shape') else 1
        return np.array([[0.1, 0.3, 0.6]] * n_samples)

class KNearestNeighbors:
    def predict_proba(self, X):
        n_samples = X.shape[0] if hasattr(X, 'shape') else 1
        return np.array([[0.2, 0.4, 0.4]] * n_samples)

m1 = OrdinalLogisticRegression()
m2 = KNearestNeighbors()

with open('models/m1_ordinal_logistic.pkl', 'wb') as f:
    pickle.dump(m1, f)

with open('models/m2_knn.pkl', 'wb') as f:
    pickle.dump(m2, f)

print("Dummy models created.")
