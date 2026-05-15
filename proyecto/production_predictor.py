"""
MÓDULO DE PREDICCIÓN EN PRODUCCIÓN
Implementa el threshold óptimo determinado por el análisis ROC-AUC

Uso:
    from production_predictor import PredictorProducción
    
    predictor = PredictorProducción('random_forest')
    predicciones = predictor.predecir(X_nuevas)
"""

import os
import json
import pickle
import numpy as np
from datetime import datetime
from preprocessing.preprocess import preprocess_pipeline


# Thresholds óptimos encontrados en análisis ROC-AUC
OPTIMAL_THRESHOLDS = {
    'Logistic Regression': 0.394,
    'Neural Network': 0.354,
    'SVM (RBF)': 0.303,
    'Random Forest': 0.343  # ⭐ MEJOR MODELO
}

# Nombres de clases
CLASS_NAMES = ['Principiante', 'Intermedio']


class PredictorProducción:
    """
    Predictor optimizado para producción con threshold óptimo.
    
    Ejemplo:
    --------
    predictor = PredictorProducción('random_forest')
    X_new = [[5, 35, 1, 0, 1, 480, 0, ...]]  # 17 features
    predicciones = predictor.predecir(X_new)
    """
    
    def __init__(self, modelo_nombre='random_forest'):
        """
        Inicializa el predictor.
        
        Parámetros:
        -----------
        modelo_nombre : str
            'random_forest', 'svm', 'logistic_regression', o 'neural_network'
        """
        self.modelo_nombre = modelo_nombre.lower()
        self.modelo = None
        self.scaler = None
        self.threshold = None
        self.X_train = None
        self.y_train = None
        
        # Mapeo de nombre a archivo y clase
        self.mapeo_modelos = {
            'random_forest': ('random_forest_model.pkl', 'Random Forest'),
            'svm': ('svm_model.pkl', 'SVM (RBF)'),
            'logistic_regression': ('logistic_regression_model.pkl', 'Logistic Regression'),
            'neural_network': ('neural_network_model.pkl', 'Neural Network'),
        }
        
        if modelo_nombre not in self.mapeo_modelos:
            raise ValueError(f"Modelo desconocido: {modelo_nombre}. Usa: {list(self.mapeo_modelos.keys())}")
        
        self._cargar_modelo()
    
    def _cargar_modelo(self):
        """Carga el modelo y scaler desde archivos guardados"""
        archivo_modelo, nombre_clase = self.mapeo_modelos[self.modelo_nombre]
        ruta_modelo = os.path.join('models', archivo_modelo)
        
        # Verificar si el modelo existe
        if not os.path.exists(ruta_modelo):
            raise FileNotFoundError(
                f"❌ No se encontró el modelo: {ruta_modelo}\n"
                f"   Por favor ejecuta: python main.py\n"
                f"   para entrenar y guardar los modelos primero."
            )
        
        # Cargar modelo
        try:
            with open(ruta_modelo, 'rb') as f:
                self.modelo = pickle.load(f)
            print(f"✓ Modelo cargado: {archivo_modelo}")
        except Exception as e:
            raise RuntimeError(f"Error al cargar modelo: {e}")
        
        # Obtener threshold óptimo
        self.threshold = OPTIMAL_THRESHOLDS[nombre_clase]
        print(f"✓ Threshold óptimo: {self.threshold:.3f}")
        print(f"  (F1-Score máximo en validación)")
    
    def predecir(self, X_nuevas, usar_threshold_optimo=True, retornar_probabilidades=False):
        """
        Realiza predicciones sobre nuevas instancias.
        
        Parámetros:
        -----------
        X_nuevas : np.ndarray o list
            Matriz de features (n_samples, 17)
        usar_threshold_optimo : bool
            Si True, usa threshold óptimo (recomendado para producción)
            Si False, usa threshold = 0.5 (default)
        retornar_probabilidades : bool
            Si True, retorna también las probabilidades
        
        Retorno:
        --------
        dict con keys:
            'predicciones': array de [0=Principiante, 1=Intermedio]
            'probabilidades': array de probabilidades (si retornar_probabilidades=True)
            'confianza': nivel de confianza (distancia al threshold)
            'timestamp': fecha/hora de predicción
        """
        X_nuevas = np.array(X_nuevas)
        
        if X_nuevas.shape[1] != 17:
            raise ValueError(f"Se esperan 17 features, recibidas {X_nuevas.shape[1]}")
        
        # Obtener probabilidades
        try:
            if hasattr(self.modelo, 'predict_proba'):
                y_proba = self.modelo.predict_proba(X_nuevas)
            else:
                raise AttributeError(f"Modelo {self.modelo_nombre} no proporciona predict_proba")
        except Exception as e:
            raise RuntimeError(f"Error en predicción: {e}")
        
        # Aplicar threshold
        if usar_threshold_optimo:
            threshold_usado = self.threshold
            y_pred = (y_proba[:, 1] >= threshold_usado).astype(int)
        else:
            threshold_usado = 0.5
            y_pred = (y_proba[:, 1] >= 0.5).astype(int)
        
        # Calcular confianza (distancia al threshold)
        confianza = np.abs(y_proba[:, 1] - threshold_usado)
        
        resultado = {
            'predicciones': y_pred,
            'clases': [CLASS_NAMES[p] for p in y_pred],
            'confianza': confianza,
            'threshold_usado': threshold_usado,
            'timestamp': datetime.now().isoformat(),
            'modelo': self.modelo_nombre,
        }
        
        if retornar_probabilidades:
            resultado['probabilidades'] = y_proba
        
        return resultado
    
    def predecir_una(self, features):
        """
        Predice una sola instancia.
        
        Parámetros:
        -----------
        features : list
            17 features del juego
        
        Retorno:
        --------
        dict con predicción, clase y confianza
        """
        resultado = self.predecir([features], retornar_probabilidades=True)
        
        # Retornar solo la primera predicción
        return {
            'prediccion': resultado['predicciones'][0],
            'clase': resultado['clases'][0],
            'confianza': resultado['confianza'][0],
            'prob_principiante': resultado['probabilidades'][0, 0],
            'prob_intermedio': resultado['probabilidades'][0, 1],
            'threshold': resultado['threshold_usado'],
            'timestamp': resultado['timestamp']
        }
    
    def comparar_thresholds(self, X_test, y_test):
        """
        Compara desempeño con diferentes thresholds.
        
        Parámetros:
        -----------
        X_test : np.ndarray
            Datos de prueba
        y_test : np.ndarray
            Etiquetas verdaderas
        
        Retorno:
        --------
        DataFrame con métricas para cada threshold
        """
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        y_proba = self.modelo.predict_proba(X_test)
        
        resultados = []
        for threshold in np.arange(0.2, 0.8, 0.05):
            y_pred = (y_proba[:, 1] >= threshold).astype(int)
            
            resultados.append({
                'threshold': threshold,
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'es_optimo': threshold == self.threshold
            })
        
        # Mostrar tabla
        print(f"\n{'Threshold':<12} | {'Precision':<12} | {'Recall':<12} | {'F1-Score':<12} | Óptimo")
        print("-" * 65)
        for r in resultados:
            marca = "⭐" if r['es_optimo'] else ""
            print(f"{r['threshold']:<12.3f} | {r['precision']:<12.4f} | {r['recall']:<12.4f} | {r['f1']:<12.4f} | {marca}")
        
        return resultados


def guardar_configuracion_produccion():
    """
    Guarda la configuración de producción con thresholds óptimos.
    """
    config = {
        'fecha_generacion': datetime.now().isoformat(),
        'modelo_recomendado': 'Random Forest',
        'thresholds_optimos': OPTIMAL_THRESHOLDS,
        'nombres_clases': CLASS_NAMES,
        'notas': [
            'Threshold óptimo para Random Forest: 0.343',
            'Produce Recall=85% para clase Intermedio',
            'Precision=45% (trade-off necesario)',
            'Usar threshold_usado=0.5 solo para baseline'
        ]
    }
    
    with open('config_produccion.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("✓ Configuración guardada en config_produccion.json")
    return config


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == '__main__':
    print("="*80)
    print("DEMOSTRACIÓN: PREDICTOR EN PRODUCCIÓN CON THRESHOLD ÓPTIMO")
    print("="*80)
    
    try:
        # Crear predictor
        print("\n1. Inicializando predictor...")
        predictor = PredictorProducción('random_forest')
        
        # Ejemplo 1: Predicción simple
        print("\n2. Predicción simple (una partida)...")
        features_ejemplo = [
            35, 20, 1, 0, 1, 480, 0, 1800, 300,  # base_time, increment
            0, 0, 1, 0, 0, 1, 0, 0  # ECO families
        ]
        
        pred = predictor.predecir_una(features_ejemplo)
        print(f"   Clase predicha: {pred['clase']}")
        print(f"   Confianza: {pred['confianza']:.3f}")
        print(f"   P(Principiante): {pred['prob_principiante']:.3f}")
        print(f"   P(Intermedio): {pred['prob_intermedio']:.3f}")
        
        # Ejemplo 2: Batch predictions
        print("\n3. Predicciones batch (100 partidas)...")
        X_batch = np.random.randn(100, 17)  # Datos simulados
        resultados = predictor.predecir(X_batch)
        
        print(f"   Total de predicciones: {len(resultados['predicciones'])}")
        print(f"   Principiantes: {sum(resultados['predicciones'] == 0)}")
        print(f"   Intermedios: {sum(resultados['predicciones'] == 1)}")
        print(f"   Confianza promedio: {resultados['confianza'].mean():.3f}")
        
        # Guardar configuración
        print("\n4. Guardando configuración...")
        guardar_configuracion_produccion()
        
        print("\n✓ Demostración completada exitosamente")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nNota: Ejecuta 'python main.py' primero para entrenar los modelos.")
