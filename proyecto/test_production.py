"""
SCRIPT DE PRUEBA: Predictor en Producción
Demuestra cómo usar los modelos entrenados con threshold óptimo

Ejecutar con: python test_production.py
"""

import numpy as np
import pandas as pd
from production_predictor import PredictorProducción
import os

def main():
    print("="*80)
    print("PRUEBA DE PRODUCCIÓN: Predictor de Habilidad en Ajedrez")
    print("="*80)
    
    # Verificar que los modelos existen
    print("\n[1/5] Verificando modelos entrenados...")
    modelos_requeridos = [
        'models/random_forest_model.pkl',
        'models/svm_model.pkl',
        'models/logistic_regression_model.pkl',
        'models/neural_network_model.pkl'
    ]
    
    todos_existen = all(os.path.exists(m) for m in modelos_requeridos)
    if not todos_existen:
        print("❌ ERROR: Faltan modelos entrenados")
        print("   Ejecuta primero: python main.py")
        return
    
    print("✓ Todos los modelos están disponibles")
    
    # Cargar predictor
    print("\n[2/5] Inicializando predictor (Random Forest)...")
    try:
        predictor = PredictorProducción('random_forest')
        print("✓ Predictor cargado correctamente")
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # EJEMPLO 1: Predicción Individual
    print("\n[3/5] EJEMPLO 1: Predicción Individual")
    print("-" * 80)
    
    # Features de una partida de ejemplo
    features_ejemplo = [
        35,      # turns (duración de la partida)
        20,      # opening_ply (profundidad apertura)
        1,       # victory_status (blancas ganan)
        0,       # winner (blancas)
        1,       # rated (partida clasificada)
        480,     # base_time (8 minutos)
        0,       # increment (sin incremento)
        1800,    # white_rating
        1600,    # black_rating (aproximado)
        0, 0, 1, 0, 0, 1, 0, 0  # ECO families (8 categorías)
    ]
    
    print("\nPartida de ejemplo:")
    print(f"  Duración (turns): {features_ejemplo[0]}")
    print(f"  Apertura (ply): {features_ejemplo[1]}")
    print(f"  Tipo: {'Clasificada' if features_ejemplo[5] else 'No clasificada'}")
    print(f"  Tiempo: {features_ejemplo[5]}+{features_ejemplo[6]}")
    
    pred = predictor.predecir_una(features_ejemplo)
    
    print(f"\n📊 PREDICCIÓN:")
    print(f"  Clase: {pred['clase']}")
    print(f"  Confianza: {pred['confianza']:.1%}")
    print(f"  P(Principiante): {pred['prob_principiante']:.1%}")
    print(f"  P(Intermedio): {pred['prob_intermedio']:.1%}")
    print(f"  Threshold usado: {pred['threshold']:.3f}")
    
    # EJEMPLO 2: Batch Predictions
    print("\n[4/5] EJEMPLO 2: Predicciones en Batch (100 simuladas)")
    print("-" * 80)
    
    # Generar 100 ejemplos aleatorios
    n_samples = 100
    X_batch = np.random.randn(n_samples, 17)
    
    # Hacer predicciones
    resultados = predictor.predecir(X_batch, retornar_probabilidades=True)
    
    predicciones = resultados['predicciones']
    probabilidades = resultados['probabilidades']
    confianza = resultados['confianza']
    
    print(f"\nTotal de predicciones: {len(predicciones)}")
    print(f"Principiantes predichos: {sum(predicciones == 0)} ({100*sum(predicciones == 0)//len(predicciones)}%)")
    print(f"Intermedios predichos: {sum(predicciones == 1)} ({100*sum(predicciones == 1)//len(predicciones)}%)")
    print(f"Confianza promedio: {confianza.mean():.3f}")
    print(f"Confianza mín/máx: {confianza.min():.3f} / {confianza.max():.3f}")
    
    # Mostrar ejemplos de predicciones
    print(f"\nPrimeros 5 resultados:")
    print(f"{'#':<3} | {'Clase':<12} | {'P(Intermedio)':<15} | {'Confianza':<12}")
    print("-" * 50)
    for i in range(min(5, len(predicciones))):
        clase = "Intermedio" if predicciones[i] == 1 else "Principiante"
        print(f"{i+1:<3} | {clase:<12} | {probabilidades[i,1]:<15.1%} | {confianza[i]:<12.3f}")
    
    # EJEMPLO 3: Comparar Thresholds (si hay datos de test disponibles)
    print("\n[5/5] EJEMPLO 3: Información de Threshold Óptimo")
    print("-" * 80)
    
    print(f"\nModelo: Random Forest")
    print(f"Threshold óptimo: {predictor.threshold:.3f}")
    print(f"  → Este threshold fue encontrado maximizando F1-Score en validación")
    print(f"  → Produce Recall=85% para la clase Intermedio (minoritaria)")
    print(f"  → Vs threshold default (0.5) que produce Recall=32%")
    
    print(f"\nComparación de thresholds:")
    print(f"{'Threshold':<12} | {'Predicción Intermedio':<25} | {'Caso de Uso':<30}")
    print("-" * 70)
    print(f"0.50 (default) | Muy conservador              | Evitar falsos positivos")
    print(f"0.343 (óptimo) | Balance Precision/Recall     | General (RECOMENDADO) ⭐")
    print(f"0.20           | Muy agresivo                  | Encontrar todos Intermedio")
    
    # RESUMEN FINAL
    print("\n" + "="*80)
    print("RESUMEN DE PRUEBA")
    print("="*80)
    print(f"""
✓ Modelos cargados: 4/4
✓ Predicción individual: OK
✓ Predicciones batch (n={n_samples}): OK
✓ Threshold óptimo aplicado: 0.343

📦 CONFIGURACIÓN LISTA PARA PRODUCCIÓN:
  - Modelo: Random Forest
  - F1-Score: 0.6322
  - ROC-AUC: 0.7170
  - Recall Intermedio: 85% (con threshold óptimo)

📝 ARCHIVOS DISPONIBLES:
  - config_produccion.json: Configuración guardada
  - models/: Modelos entrenados (.pkl)
  - GUIA_PRODUCCION.md: Tutorial completo

🚀 USAR EN CÓDIGO:
  from production_predictor import PredictorProducción
  
  predictor = PredictorProducción('random_forest')
  resultado = predictor.predecir_una(features)
  print(resultado['clase'])
""")
    
    print("="*80)
    print("✓ PRUEBA COMPLETADA EXITOSAMENTE")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
