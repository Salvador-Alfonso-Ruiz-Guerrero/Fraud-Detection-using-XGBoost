import argparse
import os
import joblib
from src.data_loader import load_data, split_data
from src.features import FeaturePipeline
from src.models import get_xgb_model
from src.evaluate import evaluate_model


def main(args):
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)

    df = load_data(args.data_path)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, target_col=args.target_col, test_size=args.test_size)

    pipeline = FeaturePipeline()
    X_train_res, y_train_res = pipeline.fit_transform(X_train, y_train)
    X_val_res, y_val_res = pipeline.transform(X_val), y_val
    X_test_scaled = pipeline.transform(X_test)

    model = get_xgb_model()
    model.fit(X_train_res, y_train_res)

    val_metrics = evaluate_model(model, X_val_res, y_val_res)
    test_metrics = evaluate_model(model, X_test_scaled, y_test)

    joblib.dump({"model": model, "pipeline": pipeline}, args.model_path)

    print("===> Validación")
    print(val_metrics)
    print("===> Test")
    print(test_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena un modelo de detección de fraude")
    parser.add_argument("--data-path", type=str, required=True, help="Ruta al CSV de datos")
    parser.add_argument("--model-path", type=str, required=True, help="Ruta de salida del modelo joblib")
    parser.add_argument("--target-col", type=str, default="Class", help="Nombre de la columna objetivo")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fracción de test")

    args = parser.parse_args()
    main(args)
