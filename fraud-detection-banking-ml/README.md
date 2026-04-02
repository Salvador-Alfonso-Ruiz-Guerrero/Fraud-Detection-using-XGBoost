# fraud-detection-banking-ml

Proyecto de detección de fraude con XGBoost.

## Estructura

- data/creditcard.csv  # dataset principal
- notebooks/01_analysis.ipynb  # EDA y storytelling
- src/data_loader.py
- src/features.py
- src/models.py
- src/evaluate.py
- src/train.py

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

```bash
python src/train.py --data-path data/creditcard.csv --model-path models/xgb_model.joblib
```
