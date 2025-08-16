# TelecomX_parte2

# TelecomX — Predicción de Cancelación (Churn)

Proyecto de analítica y Machine Learning para **predecir la cancelación de clientes** y **explicar** qué variables la impulsan. Incluye limpieza de datos, tratamiento del desbalance, comparación de modelos y análisis de importancia de variables, con código reproducible.

---

## 📌 Objetivos

1. **Modelar** la probabilidad de cancelación (*Churn = 1*).
2. **Evaluar** varios algoritmos y seleccionar el más adecuado según métricas de negocio (recall/F1 en la clase 1).
3. **Explicar** los factores clave que determinan la cancelación para proponer acciones de retención.

---

## 🗂️ Estructura sugerida del repo

```
TelecomX_parte2/
├─ data/
│  ├─ df_estandarizado.csv            # dataset tratado (sin customerID)
├─ notebooks/
│  ├─ 01_eda_preprocesamiento.ipynb
│  ├─ 02_modelado_evaluacion.ipynb
│  ├─ 03_importancia_variables.ipynb
├─ src/
│  ├─ utils.py                        # funciones auxiliares (split, métricas, plots)
│  ├─ train_eval.py                   # entrenamiento y evaluación
├─ figs/                              # gráficos exportados (PR, ROC, importancias)
├─ README.md
├─ requirements.txt
```

> Si cargas el CSV desde GitHub, usa la URL **raw**:
> `https://raw.githubusercontent.com/<usuario>/<repo>/<branch>/data/df_estandarizado.csv`

---

## 🧰 Requisitos

* Python 3.9+
* pandas, numpy, scikit-learn, imbalanced-learn, matplotlib, seaborn, jupyter

`requirements.txt`:

```
pandas
numpy
scikit-learn
imbalanced-learn
matplotlib
seaborn
jupyter
```

Instalación rápida:

```bash
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🧪 Datos

* **Filas:** 7,043 clientes
* **Target:** `Churn` (0 = no canceló, 1 = canceló)
* **Clase minoritaria:** \~26.6%

**Limpieza / estandarización**

* Se eliminó `customerID`.
* Variables categóricas → *one-hot encoding*.
* Conversión de tipos a `int`, `float` y `bool`.
* Verificación de que no queden columnas `object`.

**Multicolinealidad**

* Se detectó alta correlación (|r| > 0.8) entre **cargos** (`account.Charges.Monthly`, `account.Charges.Total`, `Cuentas_Diarias`).
* Se conservó **una** representación para evitar inestabilidad en modelos lineales.

**Split**

* `train_test_split(test_size=0.2, stratify=y, random_state=42)`

**Desbalance**

* Enfoque principal: `class_weight="balanced"` (en LogReg y RandomForest).
* Ensayos con **SMOTE** solo en *train*.

**Normalización**

* Necesaria para **Regresión Logística** y **KNN** (`StandardScaler`: *fit* en train, *transform* en test).
* **No** necesaria para **árboles/RandomForest**.

---

## ⚙️ Modelos y configuración

* **Regresión Logística**
  `LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)`

* **Random Forest**
  `RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)`

* **Decision Tree**
  `DecisionTreeClassifier(max_depth=10, class_weight="balanced", random_state=42)`

* **KNN**
  `KNeighborsClassifier(n_neighbors=5, weights="distance")`
  (selección de `k` con GridSearchCV opcional)

---

## 🧮 Evaluación (test)

Métricas: **Accuracy**, **Recall/Precision/F1** de la clase 1 (churners) y **AUC**.

| Modelo                        |  Accuracy | Recall (Churn=1) | Precisión (Churn=1) | F1 (Churn=1) |    AUC    |
| ----------------------------- | :-------: | :--------------: | :-----------------: | :----------: | :-------: |
| **LogReg (scaled, balanced)** |   0.743   |     **0.786**    |        0.510        |   **0.619**  | **0.842** |
| RandomForest (balanced)       | **0.788** |       0.484      |        0.531        |     0.548    |   0.826   |
| KNN (scaled)                  |   0.774   |       0.529      |      **0.581**      |     0.554    |   0.818   |
| DecisionTree (balanced)       |   0.733   |       0.717      |        0.498        |     0.588    |   0.768   |

**Elección operativa:** LogReg ofrece el mejor **AUC** y **F1(1)**, con **recall** alto para identificar churners.

---

## 👤 Autor

Nicolás Mora .
