# Clasificación de habilidad en ajedrez

Proyecto de machine learning para predecir si una partida pertenece a la clase Principiante o Intermedio a partir de variables derivadas de partidas de Lichess. El trabajo compara regresión logística, red neuronal, SVM y random forest, e incluye un predictor de uso en producción con umbral ajustado.

## Contexto

El dataset se procesa, se balancea con SMOTE y se divide en train/validation/test. Sobre esa base se entrenan varios modelos, se evalúan con métricas estándar y se generan gráficas y reportes para la memoria del proyecto.

## Requisitos

- Python 3.8 o superior
- Instalar dependencias con `pip install -r requirements.txt`
- Tener `data/games.csv` disponible

## Ejecución

```bash
pip install -r requirements.txt
python main.py --skip-eda --model rf
```

Opciones útiles:

```bash
python main.py
python main.py --skip-eda
python main.py --model all
python main.py --n-samples 2500
```

## Reproducir resultados

1. Instala dependencias.
2. Ejecuta `python main.py --skip-eda --model all` o la variante que quieras comparar.
3. Revisa los archivos generados al finalizar: comparación de modelos, reporte detallado, configuración de producción y gráficas en `plots/`.
4. Para probar el predictor, usa `production_predictor.py`, `test_production.py` o `api_example.py`.

Los resultados dependen de la semilla fija del proyecto y del mismo archivo `data/games.csv`.

## Estructura útil para la entrega

- `main.py`: pipeline principal
- `preprocessing/`: preparación de datos y SMOTE
- `models/`: implementaciones de los modelos
- `evaluation/`: métricas y utilidades de evaluación
- `production_predictor.py`: carga de modelo y predicción en producción
- `test_production.py` y `api_example.py`: validación de uso
- `MEMORIA_FINAL.md`: memoria del trabajo

## No subir

No es necesario incluir en la entrega los artefactos generados al ejecutar el pipeline: `plots/`, `reports/`, los `.pkl` de `models/`, `__pycache__/`, ni los reportes y configuraciones que se regeneran automáticamente.

Si necesitas rehacer todo desde cero, basta con conservar el código fuente, `requirements.txt` y `data/games.csv`.

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
