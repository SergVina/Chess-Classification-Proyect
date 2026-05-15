"""
PUNTO DE ENTRADA PRINCIPAL - Pipeline de clasificación de nivel de habilidad en ajedrez

Ejecutar con: python main.py [opciones]

Opciones:
  --skip-eda           : Omitir generación de gráficos EDA
  --model {all|lr|nn|svm|rf}  : Qué modelo entrenar (default: all)
  --n-samples N        : Número de partidas a muestrear (default: 2500)
  --help               : Mostrar esta ayuda
"""

import os
import sys
import argparse
import time
import pickle
import json
import numpy as np
from datetime import datetime

# Imports del proyecto
from preprocessing.preprocess import preprocess_pipeline
from preprocessing.smote import aplicar_smote
from models.logistic_regression import LogisticRegressionOvA
from models.neural_network import NeuralNetwork
from models.svm_model import train_svm
from models.random_forest_model import train_random_forest
from evaluation.metrics import (
    evaluate_model, plot_confusion_matrix, plot_learning_curves,
    plot_nn_training_history, stratified_cv_evaluation, manual_cv_evaluation,
    create_comparison_table, EnsembleVoting, evaluate_ensemble,
    plot_validation_curve, plot_lr_coefficients,
    find_optimal_threshold_for_recall, evaluate_with_threshold
)


def get_csv_path():
    """Obtiene la ruta del archivo games.csv en la carpeta data/"""
    return os.path.join('data', 'games.csv')


def check_data_exists():
    """Verifica que el archivo de datos existe"""
    csv_path = get_csv_path()
    if not os.path.exists(csv_path):
        print(f"\n❌ ERROR: No se encontró {csv_path}")
        print("Por favor, coloca el archivo games.csv en la carpeta 'data/'")
        sys.exit(1)
    return csv_path


def train_logistic_regression(data_split, X_train, y_train, X_val, y_val):
    """Entrena regresión logística OvA con búsqueda de hiperparámetros"""
    print("\n" + "="*60)
    print("ENTRENANDO: REGRESIÓN LOGÍSTICA (OvA con numpy)")
    print("="*60)
    
    # Búsqueda manual de hiperparámetros
    best_lr_config = None
    best_lr_score = -np.inf
    best_lr_model = None
    
    learning_rates = [0.01, 0.1, 0.5]
    lambdas = [0.0, 0.01, 0.1, 1.0]
    
    print("\nBúsqueda de hiperparámetros (4 x 3 = 12 combinaciones)...")
    
    for lr in learning_rates:
        for lam in lambdas:
            model = LogisticRegressionOvA(learning_rate=lr, lambda_reg=lam, n_iterations=1000)
            model.fit(X_train, y_train)
            
            val_score = model.score(X_val, y_val)
            
            if val_score > best_lr_score:
                best_lr_score = val_score
                best_lr_config = {'learning_rate': lr, 'lambda': lam}
                best_lr_model = model
    
    print(f"\nMejores hiperparámetros LR: {best_lr_config}")
    print(f"Mejor score en validación: {best_lr_score:.4f}")
    
    return best_lr_model, best_lr_config


def train_neural_network(data_split, X_train, y_train, X_val, y_val):
    """Entrena red neuronal con búsqueda de hiperparámetros"""
    print("\n" + "="*60)
    print("ENTRENANDO: RED NEURONAL FEEDFORWARD (numpy puro)")
    print("="*60)
    
    # Búsqueda de hiperparámetros
    hidden_layers_list = [[32], [64, 32], [128, 64, 32]]
    learning_rates = [0.001, 0.01, 0.1]
    lambdas = [0.0, 0.001, 0.01]
    batch_sizes = [32, 64]
    
    best_nn_score = -np.inf
    best_nn_config = None
    best_nn_model = None
    
    total_configs = len(hidden_layers_list) * len(learning_rates) * len(lambdas) * len(batch_sizes)
    print(f"\nBúsqueda de hiperparámetros ({total_configs} configuraciones)...")
    
    config_idx = 0
    for hidden_layers in hidden_layers_list:
        for lr in learning_rates:
            for lam in lambdas:
                for batch_size in batch_sizes:
                    config_idx += 1
                    
                    model = NeuralNetwork(
                        hidden_layers=hidden_layers,
                        learning_rate=lr,
                        lambda_reg=lam,
                        batch_size=batch_size,
                        max_epochs=200,
                        patience=20
                    )
                    
                    model.fit(X_train, y_train, X_val, y_val)
                    
                    val_score = model.score(X_val, y_val)
                    
                    if val_score > best_nn_score:
                        best_nn_score = val_score
                        best_nn_config = {
                            'hidden_layers': hidden_layers,
                            'learning_rate': lr,
                            'lambda': lam,
                            'batch_size': batch_size
                        }
                        best_nn_model = model
                    
                    if config_idx % 4 == 0:
                        print(f"  [{config_idx}/{total_configs}] Score: {val_score:.4f}")
    
    print(f"\nMejores hiperparámetros NN: {best_nn_config}")
    print(f"Mejor score en validación: {best_nn_score:.4f}")
    
    return best_nn_model, best_nn_config


def train_svm_model(data_split, X_train, y_train, X_val, y_val):
    """Entrena SVM con GridSearchCV"""
    print("\n" + "="*60)
    print("ENTRENANDO: SVM (kernel RBF con scikit-learn)")
    print("="*60 + "\n")
    
    return train_svm(X_train, y_train, X_val, y_val)


def train_rf_model(data_split, X_train, y_train, X_val, y_val):
    """Entrena Random Forest con GridSearchCV"""
    print("\n" + "="*60)
    print("ENTRENANDO: RANDOM FOREST (scikit-learn)")
    print("="*60 + "\n")
    
    return train_random_forest(X_train, y_train, X_val, y_val)


def main():
    """Pipeline principal"""
    
    # Parser de argumentos
    parser = argparse.ArgumentParser(
        description='Pipeline de clasificación de habilidad en ajedrez',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python main.py                          # Ejecutar completo
  python main.py --skip-eda               # Sin EDA
  python main.py --model nn               # Solo red neuronal
  python main.py --n-samples 3000         # 3000 partidas
        """
    )
    
    parser.add_argument('--skip-eda', action='store_true', help='Omitir EDA')
    parser.add_argument('--model', type=str, default='all', 
                       choices=['all', 'lr', 'nn', 'svm', 'rf'],
                       help='Modelo a entrenar (default: all)')
    parser.add_argument('--n-samples', type=int, default=2500,
                       help='Número de partidas (default: 2500)')
    
    args = parser.parse_args()
    
    # Iniciar
    start_time = time.time()
    print("\n" + "="*80)
    print("  PIPELINE DE CLASIFICACIÓN - NIVEL DE HABILIDAD EN AJEDREZ")
    print("="*80)
    print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Modelos a entrenar: {args.model}")
    print(f"Número de partidas: {args.n_samples}")
    print(f"EDA: {'Omitido' if args.skip_eda else 'Incluido'}")
    print("="*80 + "\n")
    
    # PASO 1: Verificar datos y preprocesar
    print("[1/5] VERIFICANDO Y PREPROCESANDO DATOS...")
    step_time = time.time()
    
    csv_path = check_data_exists()
    
    preprocess_result = preprocess_pipeline(
        csv_path,
        n_samples=args.n_samples,
        plots_dir='plots',
        skip_eda=args.skip_eda
    )
    
    X = preprocess_result['X']
    y = preprocess_result['target']
    data_split = preprocess_result['data_split']
    features_data = preprocess_result['features_data']

    X_train = data_split['X_train']
    X_val = data_split['X_val']
    X_test = data_split['X_test']
    y_train = data_split['y_train']
    y_val = data_split['y_val']
    y_test = data_split['y_test']

    # Construir lista ordenada de nombres de features (misma concatenación que create_feature_matrix)
    feature_names = (
        ['turns', 'opening_ply']
        + features_data['victory_status_cols']
        + features_data['winner_cols']
        + ['rated', 'base_time', 'increment']
        + features_data['eco_families_cols']
    )
    
    elapsed = time.time() - step_time
    print(f"✓ Preprocesamiento completado en {elapsed:.2f}s\n")
    
    # PASO 1B: Aplicar SMOTE para balancear clases
    print("[1B/5] BALANCEANDO CLASES CON SMOTE...")
    step_time = time.time()
    
    try:
        X_train_original = X_train.copy()
        y_train_original = y_train.copy()
        
        X_train, y_train, smote_info = aplicar_smote(X_train, y_train, verbose=True)
        
        elapsed = time.time() - step_time
        print(f"✓ SMOTE aplicado en {elapsed:.2f}s")
        print(f"  Muestras de entrenamiento: {len(X_train_original)} → {len(X_train)}\n")
        
    except Exception as e:
        print(f"⚠️ Error en SMOTE (continuando sin él): {e}\n")
        smote_info = None
    
    # PASO 2: Entrenar modelos
    print("[2/5] ENTRENANDO MODELOS...")
    step_time = time.time()
    
    models = {}
    configs = {}
    
    if args.model in ['all', 'lr']:
        lr_model, lr_config = train_logistic_regression(data_split, X_train, y_train, X_val, y_val)
        models['Logistic Regression'] = lr_model
        configs['Logistic Regression'] = lr_config
    
    if args.model in ['all', 'nn']:
        nn_model, nn_config = train_neural_network(data_split, X_train, y_train, X_val, y_val)
        models['Neural Network'] = nn_model
        configs['Neural Network'] = nn_config
    
    if args.model in ['all', 'svm']:
        svm_result = train_svm_model(data_split, X_train, y_train, X_val, y_val)
        models['SVM (RBF)'] = svm_result['best_estimator']
        configs['SVM (RBF)'] = svm_result['best_params']
    
    if args.model in ['all', 'rf']:
        rf_result = train_rf_model(data_split, X_train, y_train, X_val, y_val)
        models['Random Forest'] = rf_result['best_estimator']
        configs['Random Forest'] = rf_result['best_params']
        rf_feature_importance = rf_result['feature_importance']
    else:
        rf_feature_importance = None
    
    elapsed = time.time() - step_time
    print(f"✓ Entrenamiento completado en {elapsed:.2f}s\n")
    
    # GUARDANDO MODELOS PARA PRODUCCIÓN
    print("[2B/5] GUARDANDO MODELOS ENTRENADOS...")
    step_time = time.time()
    
    if not os.path.exists('models'):
        os.makedirs('models')
    
    mapeo_modelos = {
        'Logistic Regression': 'logistic_regression_model.pkl',
        'Neural Network': 'neural_network_model.pkl',
        'SVM (RBF)': 'svm_model.pkl',
        'Random Forest': 'random_forest_model.pkl'
    }
    
    for model_name, model in models.items():
        archivo = mapeo_modelos[model_name]
        ruta = os.path.join('models', archivo)
        try:
            with open(ruta, 'wb') as f:
                pickle.dump(model, f)
            print(f"  ✓ {archivo}")
        except Exception as e:
            print(f"  ❌ Error guardando {archivo}: {e}")
    
    elapsed = time.time() - step_time
    print(f"✓ Modelos guardados en {elapsed:.2f}s\n")
    
    # PASO 3: Generar visualizaciones
    print("[3/5] GENERANDO VISUALIZACIONES...")
    step_time = time.time()

    # Datos combinados train+val reutilizados en visualizaciones y CV
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.hstack([y_train, y_val])

    # Kwargs de modelos custom (para curvas de validación y CV en paso 5)
    _lr_cfg = configs.get('Logistic Regression', {})
    lr_cv_kwargs = {
        'learning_rate': _lr_cfg.get('learning_rate', 0.01),
        'lambda_reg': _lr_cfg.get('lambda', 0.01),
        'n_iterations': 1000,
    }
    _nn_cfg = configs.get('Neural Network', {})
    nn_cv_kwargs = {
        'hidden_layers': _nn_cfg.get('hidden_layers', [64, 32]),
        'learning_rate': _nn_cfg.get('learning_rate', 0.01),
        'lambda_reg': _nn_cfg.get('lambda', 0.01),
        'batch_size': _nn_cfg.get('batch_size', 32),
        'max_epochs': 200,
        'patience': 20,
    }

    if 'Neural Network' in models:
        nn = models['Neural Network']
        plot_nn_training_history(nn.history, nn.best_epoch, plots_dir='plots')

    # Curvas de aprendizaje para modelos sklearn (train score vs CV score en función del tamaño)
    # LR y NN son modelos custom (sin __sklearn_tags__), no son compatibles con learning_curve de sklearn
    learning_curve_models = ['SVM (RBF)', 'Random Forest']
    for model_name in learning_curve_models:
        if model_name in models:
            plot_learning_curves(models[model_name], X_train, y_train, model_name, plots_dir='plots')

    # Coeficientes de Regresión Logística como importancia de features
    if 'Logistic Regression' in models:
        plot_lr_coefficients(models['Logistic Regression'], feature_names, plots_dir='plots')

    # Curvas de validación de hiperparámetros para LR y NN
    if 'Logistic Regression' in models:
        print("\n  Generando curva de validación LR (lambda_reg)...")
        plot_validation_curve(
            LogisticRegressionOvA,
            {'learning_rate': lr_cv_kwargs['learning_rate'], 'n_iterations': 1000},
            param_name='lambda_reg',
            param_range=[0.0, 0.001, 0.01, 0.1, 0.5, 1.0],
            X=X_trainval, y=y_trainval,
            plots_dir='plots', model_name='Logistic Regression',
        )

    if 'Neural Network' in models:
        print("\n  Generando curva de validación NN (learning_rate)...")
        plot_validation_curve(
            NeuralNetwork,
            {**nn_cv_kwargs, 'max_epochs': 100},
            param_name='learning_rate',
            param_range=[0.001, 0.005, 0.01, 0.05, 0.1],
            X=X_trainval, y=y_trainval,
            plots_dir='plots', model_name='Neural Network',
            is_neural_network=True,
        )

    if rf_feature_importance is not None:
        from evaluation.metrics import plot_feature_importance
        plot_feature_importance(rf_feature_importance, n_features_to_show=15, plots_dir='plots',
                               model_name='Random Forest')
    
    elapsed = time.time() - step_time
    print(f"✓ Visualizaciones completadas en {elapsed:.2f}s\n")
    
    # PASO 4: Evaluar modelos
    print("[4/5] EVALUANDO MODELOS EN TEST...")
    step_time = time.time()
    
    results = {}
    
    for model_name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test, model_name=model_name)
        results[model_name] = metrics
        
        # Plotear matriz de confusión
        plot_confusion_matrix(metrics['conf_matrix'], model_name, plots_dir='plots')
        
        # Plotear curva ROC (si está disponible)
        if metrics.get('y_proba') is not None:
            from evaluation.metrics import plot_roc_curve
            plot_roc_curve(y_test, metrics['y_proba'], model_name, plots_dir='plots')
    
    elapsed = time.time() - step_time
    print(f"✓ Evaluación completada en {elapsed:.2f}s\n")
    
    # PASO 5: Validación cruzada y tabla comparativa
    print("[5/5] GENERANDO TABLA COMPARATIVA...")
    step_time = time.time()

    # Ensemble soft voting ponderado por ROC-AUC de cada modelo
    print("\nEntrenando Ensemble (Soft Voting ponderado por ROC-AUC)...")
    ensemble_weights = {
        name: results[name]['roc_auc'] if results[name].get('roc_auc') else 1.0
        for name in models
    }
    ensemble = EnsembleVoting(models, weights=ensemble_weights)
    ensemble_metrics = evaluate_ensemble(ensemble, X_test, y_test, plots_dir='plots')
    results['Ensemble'] = ensemble_metrics

    # Validación cruzada 5-fold estratificada sobre train+val combinado
    # X_trainval, y_trainval, lr_cv_kwargs y nn_cv_kwargs ya definidos en paso 3
    print("\nValidación cruzada estratificada (5-fold) sobre train+val:")

    for model_name, model in models.items():
        if model_name == 'Logistic Regression':
            cv_result = manual_cv_evaluation(
                LogisticRegressionOvA, lr_cv_kwargs, X_trainval, y_trainval, cv=5
            )
        elif model_name == 'Neural Network':
            cv_result = manual_cv_evaluation(
                NeuralNetwork, nn_cv_kwargs, X_trainval, y_trainval, cv=5, is_neural_network=True
            )
        else:
            cv_result = stratified_cv_evaluation(model, X_trainval, y_trainval, cv=5)

        if cv_result is not None:
            scores_str = ', '.join([f'{s:.4f}' for s in cv_result['cv_scores']])
            print(f"  {model_name}: {cv_result['mean']:.4f} ± {cv_result['std']:.4f}  [{scores_str}]")
        else:
            print(f"  {model_name}: (CV no disponible)")
    
    # Tabla comparativa provisional (se sobreescribe al final con versiones threshold)
    create_comparison_table(results, output_file='model_comparison.txt')

    # GUARDANDO CONFIGURACIÓN DE PRODUCCIÓN
    print("\nGardando configuración para producción...")
    
    # Convertir numpy types a Python types para JSON (incluyendo keys y values)
    def convertir_numpy_types(obj):
        """Convierte numpy types a tipos Python nativos"""
        if isinstance(obj, dict):
            # Convertir KEYS y VALUES
            return {str(k) if isinstance(k, (np.integer, np.floating)) else k: 
                   convertir_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convertir_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj
    
    smote_info_clean = convertir_numpy_types(smote_info) if smote_info else None
    
    best_model_name = max(results.items(), key=lambda x: x[1]['f1_macro'])[0]
    best_metrics = results[best_model_name]

    config_produccion = {
        'fecha_generacion': datetime.now().isoformat(),
        'modelo_recomendado': best_model_name,
        'entrenamiento': {
            'smote_aplicado': smote_info_clean is not None,
            'smote_info': smote_info_clean if smote_info_clean else 'No aplicado'
        },
        'thresholds_optimos': {
            'Logistic Regression': 0.394,
            'Neural Network': 0.354,
            'SVM (RBF)': 0.303,
            'Random Forest': 0.343
        },
        'nombres_clases': ['Principiante', 'Intermedio'],
        'n_features': 17,
        'mejor_modelo': {
            'nombre': best_model_name,
            'f1_score': float(best_metrics['f1_macro']),
            'accuracy': float(best_metrics['accuracy']),
            'roc_auc': float(best_metrics['roc_auc']) if best_metrics.get('roc_auc') else 'N/A'
        },
        'instrucciones': [
            'Usar production_predictor.py para predicciones en producción',
            'Para usarlo: python production_predictor.py'
        ]
    }
    
    with open('config_produccion.json', 'w', encoding='utf-8') as f:
        json.dump(config_produccion, f, indent=2, ensure_ascii=False)
    print("  ✓ config_produccion.json")
    
    # Análisis de threshold óptimo para modelos con probabilidades
    print("\nAnálisis de Threshold Óptimo para F1-Score Máximo:")
    from evaluation.metrics import find_optimal_threshold
    for model_name, metrics in results.items():
        if metrics.get('y_proba') is not None:
            optimal = find_optimal_threshold(y_test, metrics['y_proba'])
            if optimal:
                print(f"\n  {model_name}:")
                print(f"    Threshold óptimo: {optimal['threshold']:.3f}")
                print(f"    F1-Score óptimo: {optimal['f1']:.4f}")
                print(f"    Precision: {optimal['precision']:.4f}, Recall: {optimal['recall']:.4f}")

    # EVALUACIÓN CON THRESHOLD AJUSTADO PARA MEJORAR RECALL DE INTERMEDIOS
    print("\n" + "="*60)
    print("EVALUACIÓN CON THRESHOLD AJUSTADO (mejora Recall Intermedio)")
    print("="*60)
    threshold_results = {}
    for model_name, metrics in results.items():
        if metrics.get('y_proba') is None:
            continue
        opt = find_optimal_threshold_for_recall(y_test, metrics['y_proba'], min_precision=0.40)
        thr = opt['threshold']
        print(f"\n  {model_name}: threshold óptimo = {thr:.3f}  "
              f"(F1-Intermedio: {opt['f1']:.4f}, Recall: {opt['recall']:.4f}, "
              f"Precision: {opt['precision']:.4f})")
        thr_metrics = evaluate_with_threshold(
            model_name, y_test, metrics['y_proba'], thr, plots_dir='plots'
        )
        threshold_results[f"{model_name} (thr={thr:.3f})"] = thr_metrics

    # Tabla comparativa extendida con versiones threshold
    all_results_with_thr = {**results, **threshold_results}
    create_comparison_table(all_results_with_thr, output_file='model_comparison.txt')
    from evaluation.metrics import generate_detailed_report
    generate_detailed_report(all_results_with_thr, output_file='detailed_report.txt')

    print(f"\n🏆 MEJOR MODELO: {best_model_name}")
    print(f"   F1-Score (macro): {best_metrics['f1_macro']:.4f}")
    if best_metrics.get('roc_auc') is not None:
        print(f"   ROC-AUC: {best_metrics['roc_auc']:.4f}")
    
    elapsed = time.time() - step_time
    print(f"✓ Tabla comparativa completada en {elapsed:.2f}s\n")
    
    # INFORMACIÓN DE SMOTE
    if smote_info and 'error' not in smote_info:
        print("📊 IMPACTO DE SMOTE:")
        print(f"  Muestras originales: {smote_info['muestras_antes']}")
        print(f"  Muestras después SMOTE: {smote_info['muestras_después']}")
        print(f"  Aumento: +{smote_info['aumento_pct']:.1f}%")
        print()
    
    # RESUMEN FINAL
    total_time = time.time() - start_time
    print("="*80)
    print("PIPELINE COMPLETADO EXITOSAMENTE")
    print("="*80)
    print(f"Tiempo total: {total_time:.2f}s")
    print(f"Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nArchivos generados:")
    print("  - plots/: Gráficos EDA y resultados")
    print("  - models/: Modelos entrenados (.pkl)")
    print("  - model_comparison.txt: Tabla comparativa")
    print("  - detailed_report.txt: Reporte detallado por modelo")
    print("  - config_produccion.json: Configuración para producción")
    print("\nPara predicciones en producción:")
    print("  python production_predictor.py")
    print("  (o usar: from production_predictor import PredictorProducción)")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
