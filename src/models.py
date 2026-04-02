from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression


def get_baseline_model():
    return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)


def get_xgb_model():
    return XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42, n_jobs=-1)
