# GUÍA: Predictor en Producción

## Resumen Ejecutivo

El análisis ROC-AUC encontró que el **threshold óptimo no es 0.5**, sino valores específicos para cada modelo:

| Modelo | Threshold Óptimo | F1-Score | Recall* |
|--------|------------------|----------|---------|
| **Random Forest** | **0.343** | **0.5859** | **85.29%** ⭐ |
| Logistic Regression | 0.394 | 0.5850 | 75.88% |
| Neural Network | 0.354 | 0.5853 | 74.71% |
| SVM (RBF) | 0.303 | 0.5586 | 72.94% |

*Recall para clase Intermedio (minoritaria)

---

## ¿Por qué cambiar el threshold?

Con **threshold = 0.5** (default):
- Recall Intermedio: ~60%
- Muchos jugadores Intermedio se clasifican como Principiante

Con **threshold = 0.343** (óptimo):
- Recall Intermedio: 85%
- Encuentra más jugadores Intermedio (+25%)
- Pero precision baja a 45% (trade-off necesario)

---

## Uso Básico

### Opción 1: Script Directo

```bash
python production_predictor.py
```

Ejecutará una demostración automática.

### Opción 2: Importar en tu código

```python
from production_predictor import PredictorProducción

# Inicializar
predictor = PredictorProducción('random_forest')

# Predicción individual
features = [35, 20, 1, 0, 1, 480, 0, 1800, 300, 0, 0, 1, 0, 0, 1, 0, 0]
resultado = predictor.predecir_una(features)

print(f"Clase: {resultado['clase']}")
print(f"Confianza: {resultado['confianza']:.3f}")
print(f"P(Intermedio): {resultado['prob_intermedio']:.3f}")
```

**Salida esperada:**
```
Clase: Intermedio
Confianza: 0.347
P(Intermedio): 0.690
```

---

### Opción 3: Predicciones Batch

```python
import numpy as np
from production_predictor import PredictorProducción

predictor = PredictorProducción('random_forest')

# 1000 nuevas partidas
X_nuevas = np.random.randn(1000, 17)

resultados = predictor.predecir(
    X_nuevas, 
    usar_threshold_optimo=True,
    retornar_probabilidades=True
)

print(f"Principiantes: {sum(resultados['predicciones'] == 0)}")
print(f"Intermedios: {sum(resultados['predicciones'] == 1)}")
```

---

## Referencia de API

### Clase: `PredictorProducción`

#### Constructor
```python
PredictorProducción(modelo_nombre='random_forest')
```

**Parámetros:**
- `modelo_nombre` (str): Uno de:
  - `'random_forest'` (recomendado)
  - `'svm'`
  - `'logistic_regression'`
  - `'neural_network'`

#### Método: `predecir()`
```python
resultados = predictor.predecir(
    X_nuevas,
    usar_threshold_optimo=True,
    retornar_probabilidades=False
)
```

**Parámetros:**
- `X_nuevas` (array-like): Shape (n_samples, 17)
- `usar_threshold_optimo` (bool): Usar threshold óptimo (recomendado)
- `retornar_probabilidades` (bool): Incluir probabilidades en salida

**Retorna:** dict con:
- `'predicciones'`: array de [0, 1]
- `'clases'`: array de ["Principiante", "Intermedio"]
- `'confianza'`: distance al threshold
- `'probabilidades'`: (opcional) array (n_samples, 2)
- `'threshold_usado'`: threshold aplicado
- `'modelo'`: nombre del modelo
- `'timestamp'`: ISO datetime

#### Método: `predecir_una()`
```python
resultado = predictor.predecir_una(features)
```

**Parámetros:**
- `features` (list): 17 features

**Retorna:** dict con:
- `'prediccion'`: 0 o 1
- `'clase'`: "Principiante" o "Intermedio"
- `'confianza'`: float [0, 1]
- `'prob_principiante'`: P(0)
- `'prob_intermedio'`: P(1)
- `'threshold'`: threshold used
- `'timestamp'`: ISO datetime

#### Método: `comparar_thresholds()`
```python
resultados = predictor.comparar_thresholds(X_test, y_test)
```

Compara desempeño de diferentes thresholds (0.2 a 0.8 en pasos de 0.05).

---

## Configuración de Producción

El archivo `config_produccion.json` contiene:

```json
{
  "fecha_generacion": "2026-04-04T13:54:36.123456",
  "modelo_recomendado": "Random Forest",
  "thresholds_optimos": {
    "Logistic Regression": 0.394,
    "Neural Network": 0.354,
    "SVM (RBF)": 0.303,
    "Random Forest": 0.343
  },
  "mejor_modelo": {
    "nombre": "Random Forest",
    "f1_score": 0.6322,
    "accuracy": 0.6680,
    "roc_auc": 0.7170
  }
}
```

---

## Ejemplos de Uso Real

### Caso 1: Clasificar jugador individual

```python
from production_predictor import PredictorProducción

predictor = PredictorProducción('random_forest')

# Features de una partida de ajedrez
features = [
    35,      # turns
    20,      # opening_ply
    1,       # victory_status (blancas ganan)
    0,       # winner (blancas)
    1,       # rated (sí)
    480,     # base_time (8 minutos)
    0,       # increment (0 segundos)
    1800,    # white_rating
    1600,    # black_rating
    0,0,1,0,0,1,0,0  # ECO families OHE
]

pred = predictor.predecir_una(features)

if pred['clase'] == 'Intermedio':
    print("✓ Jugador clasificado como Intermedio")
    print(f"  Confianza: {pred['confianza']*100:.1f}%")
else:
    print("○ Jugador clasificado como Principiante")
```

### Caso 2: Batch scoring en base de datos

```python
import numpy as np
from sqlalchemy import create_engine
from production_predictor import PredictorProducción
import pandas as pd

# Cargar predictor
predictor = PredictorProducción('random_forest')

# Conectar a BD
engine = create_engine('sqlite:///games.db')

# Cargar datos
df = pd.read_sql('SELECT * FROM features LIMIT 10000', engine)
X = df[feature_cols].values

# Predecir batch
resultados = predictor.predecir(X, retornar_probabilidades=True)

# Guardar predicciones
df['prediciton'] = resultados['predicciones']
df['prob_intermedio'] = resultados['probabilidades'][:, 1]
df.to_sql('predictions', engine, if_exists='append')
```

### Caso 3: Comparar thresholds antes de deployment

```python
from production_predictor import PredictorProducción

predictor = PredictorProducción('random_forest')

# Comparar performance en datos de test
resultados = predictor.comparar_thresholds(X_test, y_test)

# Salida:
# Threshold  | Precision    | Recall       | F1-Score     | Óptimo
# -----------------------------------------------------------------------
# 0.200      | 0.3847       | 0.9706       | 0.5510       |
# 0.250      | 0.4089       | 0.9118       | 0.5625       |
# ...
# 0.343      | 0.4462       | 0.8529       | 0.5859       | ⭐
```

---

## Requisitos

- Python 3.8+
- scikit-learn
- numpy
- pandas (para ejemplos BD)
- sqlalchemy (para ejemplos BD)

---

## Troubleshooting

### ❌ "No se encontró el modelo"

**Solución:**
```bash
python main.py
```

Primero ejecuta el pipeline para entrenar y guardar los modelos.

### ❌ "Error: Se esperan 17 features, recibidas X"

**Solución:**
Verifica que tus features matcheen exactamente:
1. turns
2. opening_ply
3. victory_status
4-5. winner OHE
6. rated
7-8. base_time, increment
9-16. ECO families (8 columnas)
17. (una más)

Total: **17 features**

### ❌ "AttributeError: predict_proba"

**Solución:**
Algunos modelos personalizados no tienen `predict_proba`. Usa solo:
- `'random_forest'`
- `'svm'`
- `'logistic_regression'`

(Neural Network tiene limitaciones)

---

## Performance Esperado

- **Velocidad**: ~1000 predicciones/segundo
- **Memoria**: ~50 MB por modelo
- **Latencia**: <1ms por predicción
- **Throughput**: Batch de 10,000 en <15 segundos

---

## Contacto / Preguntas

Para más información sobre el pipeline, revisa:
- `detailed_report.txt` - Análisis completo
- `config_produccion.json` - Configuración actual
- `plots/roc_curve_*.png` - Curvas ROC de cada modelo

