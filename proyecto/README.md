# Clasificación de Habilidad en Ajedrez - Proyecto Machine Learning

**Clasificación automática del nivel de habilidad de jugadores de ajedrez (Principiante/Intermedio) usando 4 modelos de ML incluida una red neuronal desde cero con NumPy.**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![ML](https://img.shields.io/badge/ML-Advanced-brightgreen)

---

## 🎯 Inicio Rápido

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Ejecutar pipeline completo
python run_memoria.py

# 3. Revisar resultados
cat MEMORIA_FINAL.md
cat reports/model_comparison.txt
```

---

## 📋 Descripción del Proyecto

**Objetivo:** Clasificar automáticamente el nivel de habilidad de jugadores de ajedrez basándose en características de sus partidas.

**Muestras:** 2,500 partidas del dataset Lichess  
**Características:** 17 features engineerizadas  
**Clases:** 2 (Principiante/Intermedio)  
**Mejor F1-Score:** 0.644 (Logistic Regression)  
**Mejor ROC-AUC:** 0.737 (Neural Network)  

---

## 📁 Estructura del Proyecto

```
proyecto/
├── run_memoria.py                     ← ⭐ EJECUTAR ESTO
├── main.py                            ← Pipeline alternativo
├── MEMORIA_FINAL.md                   ← Análisis detallado
├── README.md                          ← Este archivo
├── requirements.txt                   ← Dependencias
│
├── data/
│   └── games.csv                      ← Dataset Lichess
│
├── preprocessing/
│   ├── preprocess.py                  ← EDA, normalización
│   └── smote.py                       ← Balanceo de clases
│
├── models/
│   ├── logistic_regression.py         ← LR OvA (NumPy)
│   ├── neural_network.py              ← MLP (NumPy)
│   ├── svm_model.py                   ← SVM (sklearn)
│   ├── random_forest_model.py         ← RF (sklearn)
│   └── *.pkl                          ← Modelos entrenados
│
├── evaluation/
│   └── metrics.py                     ← Métricas y ROC
│
├── production/
│   ├── production_predictor.py        ← API predicciones
│   ├── test_production.py             ← Tests
│   └── api_example.py                 ← Ejemplo Flask
│
├── plots/                             ← Gráficas generadas
├── reports/                           ← Reportes generados
└── .gitignore
```

## 📊 Resultados Principales

| Modelo | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------|----------|-----------|--------|----------|---------|
| **Logistic Regression** | 66.4% | 64.3% | 65.6% | **64.4%** ⭐ | 70.6% |
| Neural Network | **70.0%** | 66.0% | 63.4% | 64.0% | **73.7%** ⭐ |
| SVM (RBF) | 62.0% | 59.2% | 59.8% | 59.2% | 64.6% |
| Random Forest | 67.6% | 63.3% | 62.3% | 62.7% | 71.1% |

---

## 🚀 Instalación y Ejecución

### Requisitos
- Python 3.8+
- 4GB RAM mínimo
- Dataset `games.csv` en carpeta `data/`

### Instalación

```bash
# Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Instalar dependencias
pip install -r requirements.txt
```

### Ejecutar Pipeline

```bash
# Pipeline completo (RECOMENDADO)
python run_memoria.py

# O pipeline alternativo
python main.py

# Con opciones
python main.py --skip-eda          # Omitir EDA
python main.py --model lr          # Solo Logistic Regression
python main.py --n-samples 1500    # 1500 partidas
```

### Hacer Predicciones

```bash
# Validar sistema en producción
python production/test_production.py

# Ejecutar API Flask
python production/api_example.py

# Usar en código Python
from production.production_predictor import PredictorProducción
predictor = PredictorProducción('Random Forest')
result = predictor.predecir_una([features...])
```

## 🧠 Técnicas de Machine Learning

### 4 Modelos Implementados

**1. Logistic Regression (NumPy)** - One-vs-All Strategy
- Implementación desde cero con NumPy
- Gradient descent con L2 regularización
- Probabilidades calibradas
- ✅ Mejor F1-Score: 0.644

**2. Neural Network (NumPy)** - MLP Feedforward
- Arquitectura: [17] → [64] → [32] → [2]
- Activación: ReLU + Softmax
- Backpropagation completo
- Early stopping (paciencia=20)
- ✅ Mejor ROC-AUC: 0.737

**3. Support Vector Machine (sklearn)**
- Kernel RBF
- GridSearchCV: C × gamma (16 configs)
- Class weights balanceadas

**4. Random Forest (sklearn)**
- Ensemble de árboles
- GridSearchCV: n_estimators × max_depth × min_samples_split
- Feature importance analysis

### Técnicas Avanzadas

✅ **SMOTE:** Balanceo de clases (+31.9% muestras sintéticas)  
✅ **Class Weighting:** Penalización adaptativa para clases minoritarias  
✅ **Weighted Loss:** Softmax ponderado en NN y LR  
✅ **Validación Cruzada:** 5-fold estratificada  
✅ **Threshold Optimization:** Búsqueda de umbral óptimo por modelo  
✅ **ROC-AUC Analysis:** Curvas de rendimiento probabilístico  
✅ **Hyperparameter Tuning:** GridSearchCV para SVM y RF  

---

## 📈 Dataset y Features

### Características (17 features)

| # | Feature | Tipo | Rango | Descripción |
|----|---------|------|-------|-------------|
| 1 | turns | int | 1-500 | Total movimientos |
| 2 | opening_ply | int | 1-60 | Profundidad apertura |
| 3-5 | victory_status | binary | 0/1 | Tipo victoria (OHE) |
| 6 | winner | binary | 0/1 | Ganador |
| 7 | rated | binary | 0/1 | Partida oficial |
| 8 | base_time | int | 0-3600 | Tiempo base (seg) |
| 9 | increment | int | 0-600 | Incremento (seg) |
| 10-17 | eco_family | binary | 0/1 | Familia ECO (OHE) |

### Dataset

- **Origen:** Lichess (plataforma ajedrez online)
- **Tamaño original:** 20,058 partidas
- **Muestreo:** 2,500 (estratificado)
- **Clases:** 
  - Principiante: 1,750 (70%) → 1,500 (50% con SMOTE)
  - Intermedio: 750 (30%) → 1,500 (50% con SMOTE)
- **Split:** 60% train (1,500), 10% val (250), 30% test (750)

### Preprocesamiento Pipeline

1. Muestreo estratificado
2. Feature engineering
3. Discretización y OHE
4. Normalización StandardScaler
5. **SMOTE balancing** (+31.9% muestras)

---

## 📊 Validación y Evaluación

### Métricas Utilizadas

| Métrica | Fórmula | Propósito |
|---------|---------|----------|
| Accuracy | (TP+TN)/Total | Evaluación global |
| Precision | TP/(TP+FP) | Confiabilidad positivos |
| Recall | TP/(TP+FN) | Detección de positivos |
| F1-Score | 2×(P×R)/(P+R) | **Balance P/R** ⭐ |
| ROC-AUC | Area bajo curva | Calibración probabilística |

### Validación Cruzada

- **Estrategia:** Stratified K-Fold (k=5)
- **Beneficio:** Generalización confiable
- **Resultado:** Media ± Std < 3% (baja varianza)

---

## 📁 Archivos Generados

### Gráficas (plots/)
```
├── eda_*.png              # 6 gráficas análisis exploratorio
├── nn_training_history.png # Loss y accuracy NN
├── roc_curve_*.png         # Curvas ROC (4 modelos)
├── confusion_matrix_*.png  # Matrices confusión (4 modelos)
└── feature_importance_rf.png # Top 10 features Random Forest
```

### Reportes (reports/)
```
├── model_comparison.txt    # Tabla comparativa
├── detailed_report.txt     # Métricas detalladas por modelo
└── config_produccion.json  # Config para API producción
```

### Modelos (models/)
```
├── logistic_regression.pkl
├── neural_network.pkl
├── svm.pkl
└── random_forest.pkl
```

---

## 🎓 Resultados por Modelo

### Logistic Regression - GANADOR F1-SCORE
```
Accuracy:  66.4%
Precision: 64.3%
Recall:    65.6%
F1-Score:  64.4% ⭐
ROC-AUC:   70.6%
Threshold: 0.455
```

### Neural Network - GANADOR EXACTITUD
```
Accuracy:  70.0% ⭐
Precision: 66.0%
Recall:    63.4%
F1-Score:  64.0%
ROC-AUC:   73.7% ⭐
Threshold: 0.273
```

### Support Vector Machine
```
Accuracy:  62.0%
Precision: 59.2%
Recall:    59.8%
F1-Score:  59.2%
ROC-AUC:   64.6%
Threshold: 0.253
Recall Max: 92.9%
```

### Random Forest
```
Accuracy:  67.6%
Precision: 63.3%
Recall:    62.3%
F1-Score:  62.7%
ROC-AUC:   71.1%
Threshold: 0.343
```

---

## ⚙️ Hiperparámetros Finales

### Logistic Regression
```python
learning_rate = 0.01
alpha (regularización) = 0.01
max_iterations = 10000
```

### Neural Network
```python
hidden_layers = [64, 32]
learning_rate = 0.01
batch_size = 32
max_epochs = 500
early_stopping_patience = 20
```

### SVM
```python
kernel = 'rbf'
C = 10.0  # Encontrado por GridSearch
gamma = 0.01
class_weight = 'balanced'
```

### Random Forest
```python
n_estimators = 150
max_depth = 15
min_samples_split = 2
class_weight = 'balanced'
```

## 📖 Documentación

Para análisis detallado, consultar:

📄 **MEMORIA_FINAL.md**
- Descripción del problema
- Análisis exploratorio completo
- Técnicas implementadas
- Resultados con gráficas
- Conclusiones y recomendaciones

📊 **reports/detailed_report.txt**
- Métricas por modelo
- Reportes de clasificación
- Matrices de confusión

📋 **reports/model_comparison.txt**
- Tabla comparativa
- Thresholds óptimos
- Validación cruzada

---

## ⏱️ Reproducibilidad

- ✅ Random seeds fijados (42)
- ✅ Determinismo garantizado
- ✅ Ejecuciones múltiples dan idénticos resultados
- ✅ Codificación reproducible en Python puro

Ejecutar múltiples veces:
```bash
python run_memoria.py
# Siempre produce idénticos resultados
```

---

## 🎯 Interpretación de Resultados

### ¿Por qué Logistic Regression ganó en F1?

LR balancea mejor precision/recall con threshold óptimo (0.455), evitando sesgo a clase mayoritaria.

### ¿Por qué NN tiene mejor ROC-AUC?

Las capas no lineales generan probabilidades mejor calibradas (73.7% vs 70.6%).

### ¿Qué hace SMOTE efectivo?

```
SIN SMOTE: Modelo sesgado a predecir mayoritaria (70%)
CON SMOTE: +31.9% muestras sintéticas
RESULTADO: Recall minoritaria sube 13% (52% → 65%+)
```

### ¿Cuándo usar cada modelo?

| Caso | Recomendación |
|------|---------------|
| Máximo F1-Score | Logistic Regression |
| Máximo ROC-AUC | Neural Network (0.737) |
| Máximo Recall | SVM RBF (0.929) |
| Producción rápida | Random Forest (<1ms) |
| Interpretabilidad | Logistic Regression o RF |

---

## ⚠️ Troubleshooting

### Error: "games.csv not found"
```bash
mkdir -p data
# Copiar games.csv a data/
```

### Error: "No module named 'imblearn'"
```bash
pip install imbalanced-learn
```

### Lento o sin memoria
```bash
python run_memoria.py        # O skip EDA/modelos específicos
python main.py --model lr    # Solo LR (rápido)
```

### Verificar instalación
```bash
python -c "import numpy, pandas, sklearn, imblearn; print('✓ OK')"
```

---

## 🚀 Deployar a Producción

### Usando API Flask

```python
from flask import Flask, request, jsonify
from production.production_predictor import PredictorProducción

app = Flask(__name__)
predictor = PredictorProducción('Random Forest')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    result = predictor.predecir_una(data['features'])
    return jsonify(result)

if __name__ == '__main__':
    app.run(port=5000)
```

### Cargar Modelo Entrenado

```python
import pickle

with open('models/random_forest.pkl', 'rb') as f:
    model = pickle.load(f)

predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)
```

### Config de Producción

```python
import json

with open('reports/config_produccion.json', 'r') as f:
    config = json.load(f)

threshold = config['optimal_thresholds']['Random Forest']
```
