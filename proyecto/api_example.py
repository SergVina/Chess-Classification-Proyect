"""
API SIMPLE: Servicio de Predicción
Demuestra cómo usar el predictor como un servicio

Ejecutar con: python api_example.py
"""

from production_predictor import PredictorProducción
import json
from datetime import datetime

class ServicioPredicion:
    """Servicio simple de predicción para producción"""
    
    def __init__(self, modelo='random_forest'):
        self.predictor = PredictorProducción(modelo)
        self.predicciones_cache = []
        print(f"✓ Servicio inicializado con modelo: {modelo}")
    
    def predecir_partida(self, features_dict):
        """
        Predice el nivel de habilidad para una partida
        
        Entrada (dict):
            {
                'turns': int,
                'opening_ply': int,
                'victory_status': int,
                'winner': int,
                'rated': int,
                'base_time': int,
                'increment': int,
                'white_rating': int,
                'black_rating': int,
                'eco_families': list(8)
            }
        
        Salida:
            {
                'success': bool,
                'prediccion': str,
                'confianza': float,
                'probabilidades': dict,
                'timestamp': str,
                'message': str
            }
        """
        try:
            # Validar entrada
            features_esperadas = 17
            
            # Construir vector de features
            features = [
                features_dict.get('turns', 0),
                features_dict.get('opening_ply', 0),
                features_dict.get('victory_status', 0),
                features_dict.get('winner', 0),
                features_dict.get('rated', 0),
                features_dict.get('base_time', 0),
                features_dict.get('increment', 0),
                features_dict.get('white_rating', 0),
                features_dict.get('black_rating', 0),
            ]
            
            # Agregar ECO families
            eco_families = features_dict.get('eco_families', [0]*8)
            if len(eco_families) != 8:
                return {
                    'success': False,
                    'mensaje': f'ECO families debe tener exactamente 8 elementos, recibidas {len(eco_families)}'
                }
            
            features.extend(eco_families)
            
            if len(features) != features_esperadas:
                return {
                    'success': False,
                    'mensaje': f'Se esperan {features_esperadas} features, recibidas {len(features)}'
                }
            
            # Hacer predicción
            resultado = self.predictor.predecir_una(features)
            
            respuesta = {
                'success': True,
                'prediccion': resultado['clase'],
                'confianza': float(resultado['confianza']),
                'probabilidades': {
                    'principiante': float(resultado['prob_principiante']),
                    'intermedio': float(resultado['prob_intermedio'])
                },
                'threshold': float(resultado['threshold']),
                'timestamp': resultado['timestamp'],
                'mensaje': 'Predicción realizada exitosamente'
            }
            
            # Guardar en cache
            self.predicciones_cache.append(respuesta)
            
            return respuesta
            
        except Exception as e:
            return {
                'success': False,
                'mensaje': f'Error en predicción: {str(e)}'
            }
    
    def obtener_estadisticas(self):
        """Retorna estadísticas de predicciones realizadas"""
        if not self.predicciones_cache:
            return {'total': 0}
        
        predicciones = [p['prediccion'] for p in self.predicciones_cache if p['success']]
        confianzas = [p['confianza'] for p in self.predicciones_cache if p['success']]
        
        return {
            'total_predicciones': len(self.predicciones_cache),
            'exitosas': sum(1 for p in self.predicciones_cache if p['success']),
            'principiantes': sum(1 for p in predicciones if p == 'Principiante'),
            'intermedios': sum(1 for p in predicciones if p == 'Intermedio'),
            'confianza_promedio': sum(confianzas) / len(confianzas) if confianzas else 0,
            'ultima_prediccion': self.predicciones_cache[-1]['timestamp'] if self.predicciones_cache else None
        }


def demostrar_api():
    """Demostración de uso del servicio de API"""
    
    print("="*100)
    print("API DE PREDICCIÓN - Demostración")
    print("="*100)
    
    # Inicializar servicio
    print("\n[1/3] Inicializando servicio...")
    servicio = ServicioPredicion('random_forest')
    
    # Casos de prueba
    casos_prueba = [
        {
            'nombre': 'PartidaCorta_Rápida',
            'features': {
                'turns': 25,
                'opening_ply': 15,
                'victory_status': 1,
                'winner': 0,
                'rated': 1,
                'base_time': 300,
                'increment': 0,
                'white_rating': 1200,
                'black_rating': 1150,
                'eco_families': [1, 0, 0, 0, 0, 0, 0, 0]
            }
        },
        {
            'nombre': 'PartidaLarga_Clásica',
            'features': {
                'turns': 55,
                'opening_ply': 30,
                'victory_status': 0,
                'winner': 1,
                'rated': 1,
                'base_time': 900,
                'increment': 10,
                'white_rating': 1800,
                'black_rating': 1850,
                'eco_families': [0, 1, 0, 0, 0, 0, 0, 0]
            }
        },
        {
            'nombre': 'PartidaRápida_Blitz',
            'features': {
                'turns': 15,
                'opening_ply': 10,
                'victory_status': 1,
                'winner': 0,
                'rated': 0,
                'base_time': 180,
                'increment': 0,
                'white_rating': 1500,
                'black_rating': 1600,
                'eco_families': [0, 0, 1, 0, 0, 0, 0, 0]
            }
        }
    ]
    
    # Predecir para cada caso
    print("\n[2/3] Realizando predicciones...")
    print("-" * 100)
    
    for i, caso in enumerate(casos_prueba, 1):
        print(f"\n📌 Caso {i}: {caso['nombre']}")
        print(f"   Turns: {caso['features']['turns']}, Tiempo: {caso['features']['base_time']}s")
        
        resultado = servicio.predecir_partida(caso['features'])
        
        if resultado['success']:
            print(f"   ✓ Predicción: {resultado['prediccion']}")
            print(f"   ✓ Confianza: {resultado['confianza']:.1%}")
            print(f"   ✓ P(Principiante): {resultado['probabilidades']['principiante']:.1%}")
            print(f"   ✓ P(Intermedio): {resultado['probabilidades']['intermedio']:.1%}")
        else:
            print(f"   ❌ Error: {resultado['mensaje']}")
    
    # Estadísticas
    print("\n[3/3] Estadísticas del Servicio")
    print("-" * 100)
    
    stats = servicio.obtener_estadisticas()
    
    print(f"\n📊 ESTADÍSTICAS:")
    print(f"  Total de predicciones: {stats['total_predicciones']}")
    print(f"  Exitosas: {stats['exitosas']}")
    print(f"  Principiantes predichos: {stats['principiantes']}")
    print(f"  Intermedios predichos: {stats['intermedios']}")
    print(f"  Confianza promedio: {stats['confianza_promedio']:.3f}")
    print(f"  Última predicción: {stats['ultima_prediccion']}")
    
    # Ejemplo de respuesta JSON
    print("\n📋 EJEMPLO DE RESPUESTA JSON (para API REST):")
    print("-" * 100)
    
    respuesta_json = servicio.predecir_partida(casos_prueba[0]['features'])
    print(json.dumps(respuesta_json, indent=2, ensure_ascii=False))
    
    # Tutorial de uso
    print("\n" + "="*100)
    print("CÓMO USAR EN TU APLICACIÓN")
    print("="*100)
    print("""
1. IMPORTAR:
   from api_example import ServicioPredicion

2. INICIALIZAR:
   servicio = ServicioPredicion('random_forest')

3. PREDECIR:
   features = {
       'turns': 35,
       'opening_ply': 20,
       'victory_status': 1,
       'winner': 0,
       'rated': 1,
       'base_time': 480,
       'increment': 0,
       'white_rating': 1600,
       'black_rating': 1550,
       'eco_families': [1, 0, 0, 0, 0, 0, 0, 0]
   }
   
   resultado = servicio.predecir_partida(features)
   
   if resultado['success']:
       print(f"Nivel: {resultado['prediccion']}")
       print(f"Confianza: {resultado['confianza']:.1%}")

4. EJEMPLO CON FLASK (para crear API REST):
   
   from flask import Flask, request, jsonify
   
   app = Flask(__name__)
   servicio = ServicioPredicion('random_forest')
   
   @app.route('/predict', methods=['POST'])
   def predict():
       datos = request.json
       resultado = servicio.predecir_partida(datos)
       return jsonify(resultado)
   
   # Usar: POST http://localhost:5000/predict
   # Con JSON: {"turns": 35, "opening_ply": 20, ...}

5. GUARDAR PREDICCIONES:
   
   import sqlite3
   
   conn = sqlite3.connect('predicciones.db')
   c = conn.cursor()
   
   resultado = servicio.predecir_partida(features)
   c.execute('''INSERT INTO predicciones 
               (prediccion, confianza, timestamp) 
               VALUES (?, ?, ?)''',
            (resultado['prediccion'], 
             resultado['confianza'],
             resultado['timestamp']))
   conn.commit()
""")
    print("="*100 + "\n")

if __name__ == '__main__':
    demostrar_api()
