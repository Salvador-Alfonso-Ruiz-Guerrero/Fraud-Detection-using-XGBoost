from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


class FeaturePipeline:
    def __init__(self, scaler=None, sampler=None):
        self.scaler = scaler or StandardScaler()
        self.sampler = sampler or SMOTE(random_state=42)

    def fit_transform(self, X, y=None):
        X_scaled = self.scaler.fit_transform(X)
        if y is not None:
            X_res, y_res = self.sampler.fit_resample(X_scaled, y)
            return X_res, y_res
        return X_scaled

    def transform(self, X):
        return self.scaler.transform(X)
