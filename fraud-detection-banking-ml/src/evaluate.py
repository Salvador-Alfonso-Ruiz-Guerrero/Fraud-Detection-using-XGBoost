from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score


def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None
    report = classification_report(y, y_pred, digits=4, zero_division=0)
    matrix = confusion_matrix(y, y_pred)
    roc_auc = roc_auc_score(y, y_proba) if y_proba is not None else None
    avg_precision = average_precision_score(y, y_proba) if y_proba is not None else None

    return {
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "roc_auc": float(roc_auc) if roc_auc is not None else None,
        "average_precision": float(avg_precision) if avg_precision is not None else None,
    }
