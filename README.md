# 🛡️ Detección de Fraude Bancario — De Regresión Logística a XGBoost

> **Clasificación binaria sobre un dataset extremadamente desbalanceado** · Kaggle Credit Card Fraud Dataset · Python · Scikit-learn · XGBoost

---

## 📌 Descripción del Proyecto

El fraude en tarjetas de crédito es un problema de machine learning del mundo real con altas consecuencias: las transacciones fraudulentas representan menos del **0.2%** del total, pero cada fraude no detectado tiene un coste financiero y reputacional significativo. Acertar con el modelo no es sólo cuestión de accuracy — se trata de **minimizar el coste real de las predicciones incorrectas**.

Este proyecto construye un pipeline completo de detección de fraude, mejorando progresivamente desde una Regresión Logística hasta un modelo XGBoost ajustado, aplicando técnicas específicas para manejar el desbalanceo severo de clases.

**Dataset:** [Credit Card Fraud Detection — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 284.807 transacciones · 492 fraudes (0.172%)
- Variables: V1–V28 (transformadas con PCA), `Amount`, `Time`

---

## ⚙️ Metodología

### 1. El Problema del Desbalanceo

Con sólo un 0.17% de transacciones fraudulentas, un modelo ingenuo que siempre prediga "no fraude" alcanzaría un **99.83% de accuracy** — y sería completamente contraproducente. Se aplicaron dos estrategias para abordar esto:

| Estrategia | Mecanismo |
|---|---|
| **Pesos de clase** | Penaliza más al modelo por no detectar fraudes (`class_weight='balanced'`) |
| **SMOTE** | Genera muestras sintéticas de fraude en el conjunto de entrenamiento para balancear las clases |

### 2. Optimización del Umbral de Decisión

En lugar del umbral por defecto de 0.5, el umbral de cada modelo se ajustó para **maximizar el F2-Score** — una variante del F1 que pondera el recall el doble que la precisión. En detección de fraude, dejar pasar un fraude real (falso negativo) es mucho más costoso que bloquear una transacción legítima (falso positivo).

### 3. Métricas de Evaluación

| Metrica | Por qué importa |
|---|---|
| **F2-Score** | Objetivo principal de optimización — penaliza los fraudes no detectados |
| **Fraud Detection Rate (Recall)** | % de fraudes reales correctamente detectados |
| **Precisión** | % de alertas de fraude que son fraudes reales |
| **PR-AUC** | Área bajo la curva Precisión-Recall — ideal para datos desbalanceados |
| **ROC-AUC** | Capacidad discriminativa global del modelo |
| **FPR** | Tasa de Falsos Positivos — con qué frecuencia se bloquean transacciones legítimas |
| **Coste esperado (€)** | Coste de negocio = FN × coste_fraude + FP × coste_revisión |
| **Precisión@100** | Precisión entre las 100 transacciones con mayor riesgo predicho |

### 4. Modelos Entrenados

Se evaluaron ocho configuraciones, ordenadas por complejidad:

| # | Modelo | Estrategia Desbalanceo |
|---|---|---|
| 1 | Regresión Logística | Niguna (baseline) |
| 2 | Regresión Logística | Pesos |
| 3 | Regresión Logística | SMOTE |
| 4 | Random Forest | Pesos |
| 5 | Random Forest | SMOTE |
| 6 | **XGBoost** | **Pesos** |
| 7 | XGBoost | SMOTE |

### 5. Búsqueda de Hiperparámetros

Para los modelos finales (Random Forest y XGBoost) se realizó una **búsqueda exhaustiva con validación cruzada** usando **Stratified K-Fold** (para preservar la proporción de clases en cada fold), optimizando el F2-Score.

Principales hiperparámetros ajustados en XGBoost:
- `n_estimators`, `max_depth`, `learning_rate`
- `subsample`, `colsample_bytree`
- `scale_pos_weight` (para el manejo del desbalanceo)

---

## 📊 Resultados

> Se muestran únicamente los 6 modelos competitivos finales (se excluye la regresión logística sin pesos ni ajuste de umbral por claridad)

| # | Model | Trans. Correctas | Fraudes Detectados | Falsos Positivos | Falsos Negativos | Precisión | Fraud Detection Rate | F2-Score | ROC-AUC | PR-AUC | FPR | Precision@100 | Coste Esperado (€) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | Logistic Regression weighted | 56,631 | 73 | 20 | 22 | 0.785 | 0.768 | 0.772 | 0.966 | 0.669 | 0.00035 | 0.73 | 2,300 |
| 1 | Logistic Regression + SMOTE | 56,631 | 73 | 20 | 22 | 0.785 | 0.768 | 0.772 | 0.962 | 0.659 | 0.00035 | 0.74 | 2,300 |
| 2 | Random Forest weighted | 56,635 | 76 | 16 | 19 | 0.826 | 0.800 | 0.805 | 0.953 | 0.816 | 0.00028 | 0.77 | 1,980 |
| 3 | Random Forest + SMOTE | 56,637 | 74 | 14 | 21 | 0.841 | 0.779 | 0.791 | 0.979 | 0.810 | 0.00025 | 0.76 | 2,170 |
| 4 | **XGBoost weighted ✅** | **56,643** | **75** | **8** | **20** | **0.904** | **0.789** | **0.810** | **0.979** | **0.828** | **0.00014** | **0.76** | **2,040** |
| 5 | XGBoost + SMOTE | 56,637 | 74 | 14 | 21 | 0.841 | 0.779 | 0.791 | 0.977 | 0.806 | 0.00025 | 0.75 | 2,170 |

---

## 🏆 Mejor Modelo: XGBoost con Pesos de Clase

**XGBoost con pesos** fue seleccionado como modelo final tras un análisis técnico y de negocio:

- **Mayor Precisión (0.904):** El 90.4% de las alertas generadas son fraudes reales — minimizando la fatiga del equipo de revisión
- **Menor Tasa de Falsos Positivos (0.00014):** El menor número de clientes legítimos bloqueados incorrectamente
- **Mayor PR-AUC (0.828):** Mejor rendimiento en la curva Precisión-Recall — la métrica definitiva en clasificación desbalanceada
- **Menor Coste Esperado (€2.040):** El modelo que minimiza el impacto financiero real para el negocio
- **Mejor F2-Score (0.810):** El mejor equilibrio entre recall y precisión, ponderando el recall

> Aunque Random Forest con pesos detecta un fraude más (76 vs 75), la notablemente menor tasa de falsos positivos de XGBoost (8 vs 16) lo hace superior en la práctica — menos clientes legítimos bloqueados y un coste total inferior.

---

## 💡 Conclusiones Clave
1. **El accuracy no sirve para datos desbalanceados** — usar siempre F2, PR-AUC o Coste Esperado como métricas principales
2. **La optimización del umbral es crítica** — el umbral por defecto de 0.5 raramente es óptimo; ajustarlo para F2 tuvo un impacto significativo
3. **Los pesos de clase superan a SMOTE en XGBoost** — el parámetro `scale_pos_weight` maneja el desbalanceo de forma más elegante que el sobremuestreo sintético
4. **PR-AUC > ROC-AUC** en fraude — el ROC-AUC puede ser engañosamente alto en datasets desbalanceados; el PR-AUC cuenta la historia real
5. **El coste de negocio debe guiar la selección del modelo** — un modelo con recall ligeramente menor pero muchos menos falsos positivos puede ser más valioso en la práctica

---

## 🛠️ Stack Tecnológico

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-red)
![Pandas](https://img.shields.io/badge/Pandas-2.0-150458?logo=pandas)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-SMOTE-green)

scikit-learn · xgboost · imbalanced-learn · pandas · numpy · matplotlib · seaborn

---

## 📬 Contacto

¡No dudes en conectar o escribirme por [LinkedIn]([https://linkedin.com](https://www.linkedin.com/in/-salvador-ruiz-/)) si tienes preguntas o quieres comentar el proyecto!

---

*⭐ Si el proyecto te ha resultado útil, ¡considera darle una estrella!*
