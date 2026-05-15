"""
SCRIPT AVANZADO: Predicciones en Producción con Datos Reales
Carga datos del dataset de prueba y hace predicciones

Ejecutar con: python predict_on_real_data.py
"""

import numpy as np
import pandas as pd
import pickle
import os
from production_predictor import PredictorProducción
from sklearn.metrics import confusion_matrix, classification_report
from datetime import datetime

def load_test_data():
    """Carga datos de prueba del dataset preprocesado"""
    try:
        # Intentar cargar datos procesados
        with open('data/test_data.pkl', 'rb') as f:
            data = pickle.load(f)
        return data
    except:
        print("⚠️ Datos de prueba no encontrados, generando datos sintéticos...")
        return None

def generate_sample_data(n_samples=50):
    """Genera datos de ejemplo realistas"""
    print(f"\n📊 Generando {n_samples} partidas de ejemplo...")
    
    # Simulación realista de features
    np.random.seed(42)
    X = np.random.randn(n_samples, 17)
    
    # Normalizar a rangos realistas
    X[:, 0] = np.abs(X[:, 0] * 20 + 40).astype(int)  # turns: 20-60
    X[:, 1] = np.abs(X[:, 1] * 15 + 20).astype(int)  # opening_ply: 5-35
    X[:, 2:6] = np.random.randint(0, 2, (n_samples, 4))  # variables binarias
    X[:, 6] = np.abs(X[:, 6] * 500 + 480).astype(int)  # base_time
    X[:, 7] = np.abs(X[:, 7] * 100).astype(int)  # increment
    X[:, 8:] = np.random.randn(n_samples, 9)  # ratings y categorías
    
    return X

def main():
    print("="*100)
    print("PREDICCIONES EN PRODUCCIÓN - Datos Reales")
    print("="*100)
    
    # 1. Cargar predictor
    print("\n[1/4] Inicializando predictor...")
    try:
        predictor = PredictorProducción('random_forest')
        print("✓ Random Forest cargado (ROC-AUC: 0.7170)")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("   Ejecuta primero: python main.py")
        return
    
    # 2. Cargar o generar datos
    print("\n[2/4] Cargando datos...")
    test_data = load_test_data()
    
    if test_data is not None:
        X_test = test_data.get('X_test', None)
        y_test = test_data.get('y_test', None)
        print(f"✓ Datos reales cargados: {X_test.shape[0]} muestras")
    else:
        X_test = generate_sample_data(n_samples=50)
        y_test = None
        print(f"✓ Datos de ejemplo generados: {X_test.shape[0]} muestras")
    
    # 3. Hacer predicciones
    print("\n[3/4] Realizando predicciones...")
    
    resultados = predictor.predecir(
        X_test,
        usar_threshold_optimo=True,
        retornar_probabilidades=True
    )
    
    y_pred = resultados['predicciones']
    y_proba = resultados['probabilidades']
    clases = resultados['clases']
    confianza = resultados['confianza']
    
    print(f"✓ {len(y_pred)} predicciones realizadas")
    
    # 4. Análisis de resultados
    print("\n[4/4] Análisis de Resultados")
    print("-" * 100)
    
    # Estadísticas generales
    n_principiante = sum(y_pred == 0)
    n_intermedio = sum(y_pred == 1)
    
    print(f"\n📊 DISTRIBUCIÓN DE PREDICCIONES:")
    print(f"  Principiante: {n_principiante} ({100*n_principiante//len(y_pred)}%) 🔵")
    print(f"  Intermedio:   {n_intermedio} ({100*n_intermedio//len(y_pred)}%) 🟡")
    
    print(f"\n📈 ESTADÍSTICAS DE CONFIANZA:")
    print(f"  Promedio: {confianza.mean():.3f}")
    print(f"  Mínimo:   {confianza.min():.3f}")
    print(f"  Máximo:   {confianza.max():.3f}")
    print(f"  Mediana:  {np.median(confianza):.3f}")
    print(f"  Desv. Est: {confianza.std():.3f}")
    
    print(f"\n📊 PROBABILIDADES (P(Intermedio)):")
    print(f"  Promedio: {y_proba[:, 1].mean():.3f}")
    print(f"  Mínimo:   {y_proba[:, 1].min():.3f}")
    print(f"  Máximo:   {y_proba[:, 1].max():.3f}")
    
    # Tabla de ejemplos
    print(f"\n📋 PRIMEROS 10 RESULTADOS:")
    print(f"{'#':<3} | {'Clase':<12} | {'P(Intermedio)':<15} | {'Confianza':<12} | {'Clasificación':<15}")
    print("-" * 70)
    
    for i in range(min(10, len(y_pred))):
        clase_nombre = "Intermedio" if y_pred[i] == 1 else "Principiante"
        marca = "✓" if confianza[i] > 0.3 else "!"
        print(f"{i+1:<3} | {clase_nombre:<12} | {y_proba[i, 1]:<15.1%} | {confianza[i]:<12.3f} | {marca:<15}")
    
    # Si tenemos datos reales con etiquetas
    if y_test is not None:
        print(f"\n" + "="*100)
        print("COMPARACIÓN CON DATOS REALES")
        print("="*100)
        
        # Convertir a binario si es multi-clase
        if len(np.unique(y_test)) <= 2:
            y_test_binary = y_test
        else:
            # Si hay 3 clases, usar solo Principiante vs Intermedio
            mask = y_test < 2
            y_test_binary = y_test[mask]
            y_pred = y_pred[mask]
            X_test = X_test[mask]
        
        # Matriz de confusión
        cm = confusion_matrix(y_test_binary, y_pred)
        
        print(f"\n📊 MATRIZ DE CONFUSIÓN:")
        print(f"{'':12} | {'Pred: P':<12} | {'Pred: I':<12}")
        print("-" * 40)
        print(f"Real: P    | {cm[0,0]:<12} | {cm[0,1]:<12}")
        print(f"Real: I    | {cm[1,0]:<12} | {cm[1,1]:<12}")
        
        # Métricas
        accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
        print(f"\n📈 MÉTRICAS:")
        print(f"  Accuracy: {accuracy:.1%}")
        
        if cm.sum() > 0:
            print(f"\n  Recall Principiante: {cm[0,0]/(cm[0,0]+cm[0,1]):.1%}")
            print(f"  Recall Intermedio:   {cm[1,1]/(cm[1,0]+cm[1,1]):.1%}")
            print(f"  Precision Intermedio: {cm[1,1]/(cm[0,1]+cm[1,1]):.1%}")
        
        # Classification report
        print(f"\n📋 CLASSIFICATION REPORT:")
        print(classification_report(y_test_binary, y_pred, 
                                   target_names=['Principiante', 'Intermedio']))
    
    # RESUMEN FINAL
    print("\n" + "="*100)
    print("RESUMEN DE PREDICCIÓN EN PRODUCCIÓN")
    print("="*100)
    print(f"""
✓ Modelo: Random Forest
✓ Threshold óptimo: 0.343 (aplicado)
✓ ROC-AUC: 0.7170

📊 RESULTADOS:
  - Predicciones: {len(y_pred)}
  - Principiante: {n_principiante} ({100*n_principiante//len(y_pred)}%)
  - Intermedio:   {n_intermedio} ({100*n_intermedio//len(y_pred)}%)
  - Confianza promedio: {confianza.mean():.3f}

🎯 INTERPRETACIÓN:
  - Threshold 0.343 reduce falsos negativos de Intermedio
  - Confianza > 0.3 indica predicción firme
  - Para casos ambiguos (confianza < 0.1), considerar revisión manual

🚀 PRÓXIMOS PASOS:
  1. Integrar predictor en aplicación web/API
  2. Guardar predicciones en base de datos
  3. Monitorear accuracy en producción
  4. Reentrenar mensualmente con nuevos datos

✓ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""")
    print("="*100 + "\n")

if __name__ == '__main__':
    main()
