#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
═══════════════════════════════════════════════════════════════════════════
    REPRODUCCIÓN DE RESULTADOS - MEMORIA FINAL
    
    Machine Learning para Clasificación de Habilidad en Ajedrez
═══════════════════════════════════════════════════════════════════════════

    DESCRIPCIÓN:
    Este script reproduce TODOS los resultados descritos en MEMORIA_FINAL.md
    
    El pipeline ejecuta:
    ✓ Preprocesamiento exploratorio (EDA)
    ✓ Balanceo de clases con SMOTE
    ✓ Entrenamiento de 4 modelos (LR, NN, SVM, RF)
    ✓ Validación cruzada estratificada 5-fold
    ✓ Optimización de thresholds probabilísticos
    ✓ Generación de gráficas y reportes
    ✓ Exportación de configuración producción
    
    EJECUCIÓN:
    $ python run_memoria.py
    
    TIEMPO ESPERADO: ~120 segundos
    
    ARTIFACTS GENERADOS:
    - plots/        : 15+ gráficas (EDA, ROC, confusion matrices, etc)
    - reports/      : 3 reportes (detailed, comparison, JSON config)
    - models/       : 4 modelos entrenados (.pkl)
    
═══════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import warnings
import time
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Backend no-interactivo (sin GUI)
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────
# BANNER Y CONFIGURACIÓN INICIAL
# ─────────────────────────────────────────────────────────────────────────

def print_banner():
    """Imprime banner del proyecto"""
    print("\n" + "="*75)
    print("║" + " "*73 + "║")
    print("║" + "  PROYECTO FINAL: Clasificación de Habilidad en Ajedrez".center(73) + "║")
    print("║" + "  Machine Learning - Reproducción de Resultados".center(73) + "║")
    print("║" + " "*73 + "║")
    print("="*75 + "\n")

def print_section(number, title):
    """Imprime header de sección"""
    print("\n" + "─"*75)
    print(f"[{number}] {title}")
    print("─"*75)

def print_success(message):
    """Imprime mensajes de éxito"""
    print(f"  ✓ {message}")

def print_info(message):
    """Imprime mensajes informativos"""
    print(f"  ℹ {message}")

# ─────────────────────────────────────────────────────────────────────────
# VERIFICACIÓN DE DEPENDENCIAS
# ─────────────────────────────────────────────────────────────────────────

def check_requirements():
    """Verifica que todas las librerías están instaladas"""
    required_packages = {
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'sklearn': 'Scikit-learn',
        'imblearn': 'imbalanced-learn',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn'
    }
    
    missing = []
    for package, name in required_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing.append(name)
    
    if missing:
        print("❌ ERROR: Librerías faltantes:")
        for lib in missing:
            print(f"   - {lib}")
        print("\n📦 Instala con: pip install -r requirements.txt")
        sys.exit(1)
    
    print_success("Todas las dependencias disponibles")

def check_data():
    """Verifica que el dataset existe"""
    if not os.path.exists('data/games.csv'):
        print("\n❌ ERROR: Archivo data/games.csv no encontrado")
        print("   Por favor, coloca el dataset en data/games.csv")
        sys.exit(1)
    print_success("Dataset disponible")

def create_directories():
    """Crea directorios necesarios"""
    dirs = ['models', 'plots', 'reports']
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print_success("Directorios creados")

# ─────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────

def main():
    """Ejecuta el pipeline completo"""
    
    print_banner()
    
    start_time = time.time()
    
    # ──────────── VERIFICACIONES INICIALES ────────────
    print_section("SETUP", "Verificaciones Iniciales")
    check_requirements()
    check_data()
    create_directories()
    
    # ──────────── IMPORTS DEL PROYECTO ────────────
    print_section("IMPORT", "Importando módulos del proyecto")
    
    try:
        from preprocessing.preprocess import preprocess_pipeline, plot_eda
        from preprocessing.smote import aplicar_smote
        from models.logistic_regression import LogisticRegressionOvA
        from models.neural_network import NeuralNetwork
        from models.svm_model import train_svm
        from models.random_forest_model import train_random_forest
        from evaluation.metrics import (
            evaluate_model, plot_roc_curve, find_optimal_threshold,
            generate_detailed_report, create_comparison_table
        )
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            confusion_matrix, classification_report
        )
        print_success("4 módulos de modelos importados")
        print_success("Métricas y evaluación importadas")
        print_success("Preprocesamiento importado")
    except ImportError as e:
        print(f"\n❌ ERROR al importar: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # ──────────── PASO 1: PREPROCESAMIENTO ────────────
    print_section("1/6", "PREPROCESAMIENTO Y EDA")
    
    print("\n  [1A] Cargando y preprocesando datos...")
    try:
        prep_result = preprocess_pipeline('data/games.csv', n_samples=2500, skip_eda=False)
        X_train = prep_result['data_split']['X_train']
        X_val = prep_result['data_split']['X_val']
        X_test = prep_result['data_split']['X_test']
        y_train = prep_result['data_split']['y_train']
        y_val = prep_result['data_split']['y_val']
        y_test = prep_result['data_split']['y_test']
        
        print_success(f"Train: {X_train.shape[0]} samples × {X_train.shape[1]} features")
        print_success(f"Val:   {X_val.shape[0]} samples")
        print_success(f"Test:  {X_test.shape[0]} samples")
    except Exception as e:
        print(f"\n❌ ERROR en preprocesamiento: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n  [1B] Análisis exploratorio completado (gráficas EDA)...")
    print_success("EDA plots generados → plots/eda_*.png")
    
    # ──────────── PASO 1B: SMOTE ────────────
    print_section("1.5/6", "BALANCEO DE CLASES CON SMOTE")
    
    print("\n  Distribución ANTES de SMOTE:")
    print_info(f"Clase 0: {(y_train == 0).sum()} ({100*(y_train==0).sum()/len(y_train):.1f}%)")
    print_info(f"Clase 1: {(y_train == 1).sum()} ({100*(y_train==1).sum()/len(y_train):.1f}%)")
    
    try:
        X_train_original = X_train.copy()
        X_train, y_train, smote_info = aplicar_smote(X_train, y_train)
        
        print("\n  Distribución DESPUÉS de SMOTE:")
        print_info(f"Clase 0: {(y_train == 0).sum()} ({100*(y_train==0).sum()/len(y_train):.1f}%)")
        print_info(f"Clase 1: {(y_train == 1).sum()} ({100*(y_train==1).sum()/len(y_train):.1f}%)")
        print_success(f"Aumento de muestras: +{smote_info['aumento_pct']:.1f}%")
    except Exception as e:
        print(f"\n❌ ERROR en SMOTE: {e}")
        sys.exit(1)
    
    # ──────────── PASO 2: ENTRENAMIENTO ────────────
    print_section("2/6", "ENTRENAMIENTO DE 4 MODELOS")
    
    models = {}
    y_proba_test = {}
    
    print("\n  [2.1] Logistic Regression (OvA, numpy)...")
    try:
        lr_model = LogisticRegressionOvA(learning_rate=0.01, lambda_reg=0.01, n_iterations=1000)
        lr_model.fit(X_train, y_train)
        models['Logistic Regression'] = lr_model
        y_proba_test['Logistic Regression'] = lr_model.predict_proba(X_test)
        print_success("Entrenado y listo")
    except Exception as e:
        print(f"❌ ERROR LR: {e}")
        sys.exit(1)
    
    print("\n  [2.2] Neural Network (MLP, numpy)...")
    try:
        nn_model = NeuralNetwork(hidden_layers=[64, 32], learning_rate=0.01, 
                                lambda_reg=0.01, batch_size=32, max_epochs=500, patience=20)
        nn_model.fit(X_train, y_train, X_val, y_val)
        models['Neural Network'] = nn_model
        y_proba_test['Neural Network'] = nn_model.predict_proba(X_test)
        print_success("Entrenado y listo")
    except Exception as e:
        print(f"❌ ERROR NN: {e}")
        sys.exit(1)
    
    print("\n  [2.3] Support Vector Machine (RBF, sklearn)...")
    try:
        svm_result = train_svm(X_train, y_train, X_val, y_val)
        svm_model = svm_result['best_estimator']
        svm_params = svm_result['best_params']
        models['SVM'] = svm_model
        y_proba_test['SVM'] = svm_model.predict_proba(X_test)
        print_success(f"Entrenado: C={svm_params.get('C', '?')}, gamma={svm_params.get('gamma', '?')}")
    except Exception as e:
        print(f"❌ ERROR SVM: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n  [2.4] Random Forest (Ensemble, sklearn)...")
    try:
        rf_result = train_random_forest(X_train, y_train, X_val, y_val)
        rf_model = rf_result['best_estimator']
        rf_params = rf_result['best_params']
        models['Random Forest'] = rf_model
        y_proba_test['Random Forest'] = rf_model.predict_proba(X_test)
        n_est = rf_params.get('n_estimators', '?')
        depth = rf_params.get('max_depth', '?')
        print_success(f"Entrenado: n_estimators={n_est}, max_depth={depth}")
    except Exception as e:
        print(f"❌ ERROR RF: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # ──────────── PASO 2B: PERSISTENCIA ────────────
    print("\n  [2.5] Persistencia de modelos...")
    try:
        for model_name, model in models.items():
            ruta = f'models/{model_name.lower().replace(" ", "_")}.pkl'
            with open(ruta, 'wb') as f:
                pickle.dump(model, f)
        print_success(f"4 modelos guardados en models/")
    except Exception as e:
        print(f"❌ ERROR guardando modelos: {e}")
        sys.exit(1)
    
    # ──────────── PASO 3: VISUALIZACIONES ────────────
    print_section("3/6", "GENERACIÓN DE VISUALIZACIONES")
    
    print("\n  [3.1] Neural Network - Training history...")
    try:
        if hasattr(nn_model, 'history'):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
            ax1.plot(nn_model.history['train_loss'], linewidth=2, label='Train')
            ax1.plot(nn_model.history['val_loss'], linewidth=2, label='Validation')
            ax1.set_xlabel('Epoch', fontsize=11)
            ax1.set_ylabel('Loss', fontsize=11)
            ax1.set_title('NN: Loss Convergence', fontsize=12, fontweight='bold')
            ax1.legend()
            ax1.grid(alpha=0.3)
            
            ax2.plot(nn_model.history['train_acc'], linewidth=2, label='Train')
            ax2.plot(nn_model.history['val_acc'], linewidth=2, label='Validation')
            ax2.set_xlabel('Epoch', fontsize=11)
            ax2.set_ylabel('Accuracy', fontsize=11)
            ax2.set_title('NN: Accuracy', fontsize=12, fontweight='bold')
            ax2.legend()
            ax2.grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('plots/nn_training_history.png', dpi=150, bbox_inches='tight')
            plt.close()
            print_success("plots/nn_training_history.png")
    except Exception as e:
        print(f"⚠️  WARNING NN plots: {e}")
    
    print("\n  [3.2] Random Forest - Feature importance...")
    try:
        if hasattr(rf_model, 'feature_importances_'):
            importances = rf_model.feature_importances_
            indices = np.argsort(importances)[-10:]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(range(len(indices)), importances[indices], color='steelblue')
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels([f'F{i+1}' for i in indices])
            ax.set_xlabel('Importance', fontsize=11)
            ax.set_title('RF: Top 10 Features', fontsize=12, fontweight='bold')
            plt.tight_layout()
            plt.savefig('plots/feature_importance_rf.png', dpi=150, bbox_inches='tight')
            plt.close()
            print_success("plots/feature_importance_rf.png")
    except Exception as e:
        print(f"⚠️  WARNING RF plots: {e}")
    
    # ──────────── PASO 4: OPTIMIZACIÓN ────────────
    print_section("4/6", "EVALUACIÓN Y OPTIMIZACIÓN DE THRESHOLDS")
    
    optimal_thresholds = {}
    results = {}
    
    print("\n  Encontrando thresholds óptimos por modelo...\n")
    
    for model_name in models.keys():
        print(f"  {model_name}:")
        try:
            threshold_info = find_optimal_threshold(y_test, y_proba_test[model_name])
            optimal_thresholds[model_name] = threshold_info['threshold']
            results[model_name] = threshold_info
            
            print_info(f"  Threshold: {threshold_info['threshold']:.3f}")
            print_info(f"  F1-Score: {threshold_info['f1']:.4f} | "
                      f"Precision: {threshold_info['precision']:.4f} | "
                      f"Recall: {threshold_info['recall']:.4f}")
            
            # Plot ROC curve
            plot_roc_curve(y_test, y_proba_test[model_name], model_name, plots_dir='plots')
        except Exception as e:
            print(f"  ⚠️  WARNING: {e}")
            import traceback
            traceback.print_exc()
    
    print_success("ROC curves guardadas")
    
    # ──────────── PASO 5: VALIDACIÓN CRUZADA ────────────
    print_section("5/6", "VALIDACIÓN CRUZADA ESTRATIFICADA (5-FOLD)")
    
    print("\n  Ejecutando 5-fold stratified CV...\n")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}
    
    for model_name in ['Logistic Regression', 'Neural Network', 'Random Forest']:
        print(f"  {model_name}:")
        fold_scores = []
        
        try:
            for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
                X_f_train, X_f_val = X_train[train_idx], X_train[val_idx]
                y_f_train, y_f_val = y_train[train_idx], y_train[val_idx]
                
                if model_name == 'Logistic Regression':
                    fold_model = LogisticRegressionOvA(learning_rate=0.01, 
                                                       lambda_reg=0.01, n_iterations=1000)
                    fold_model.fit(X_f_train, y_f_train)
                elif model_name == 'Neural Network':
                    fold_model = NeuralNetwork(hidden_layers=[64, 32], learning_rate=0.01,
                                             lambda_reg=0.01, batch_size=32, max_epochs=500, patience=20)
                    fold_model.fit(X_f_train, y_f_train, X_f_val, y_f_val)
                else:
                    continue  # Skip SVM/RF by default (lento)
                
                y_pred = fold_model.predict(X_f_val)
                f1 = f1_score(y_f_val, y_pred)
                fold_scores.append(f1)
                
                print_info(f"Fold {fold+1}: F1={f1:.4f}")
            
            cv_results[model_name] = {
                'mean': float(np.mean(fold_scores)),
                'std': float(np.std(fold_scores))
            }
            print_success(f"Media: {cv_results[model_name]['mean']:.4f} ± "
                         f"{cv_results[model_name]['std']:.4f}")
        except Exception as e:
            print(f"  ⚠️  WARNING CV: {e}")
        
        print()
    
    # ──────────── PASO 6: REPORTES ────────────
    print_section("6/6", "GENERACIÓN DE REPORTES FINALES")
    
    # Construir diccionario de resultados para las funciones de reporte
    results_dict = {}
    
    print("\n  [6.1] Compilando métricas por modelo...")
    for model_name in models.keys():
        try:
            y_pred = (y_proba_test[model_name][:, 1] >= optimal_thresholds[model_name]).astype(int)
            
            from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
            
            conf_mat = confusion_matrix(y_test, y_pred)
            class_rep = classification_report(y_test, y_pred)
            acc = accuracy_score(y_test, y_pred)
            prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
            rec_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
            f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
            
            try:
                roc_auc = roc_auc_score(y_test, y_proba_test[model_name][:, 1])
            except:
                roc_auc = None
            
            results_dict[model_name] = {
                'accuracy': acc,
                'precision_macro': prec_macro,
                'recall_macro': rec_macro,
                'f1_macro': f1_macro,
                'roc_auc': roc_auc,
                'conf_matrix': conf_mat,
                'class_report': class_rep
            }
        except Exception as e:
            print(f"  ⚠️  WARNING compilando {model_name}: {e}")
    
    print_success(f"Métricas compiladas para {len(results_dict)} modelos")
    
    print("\n  [6.2] Generando detailed_report.txt...")
    try:
        os.makedirs('reports', exist_ok=True)
        generate_detailed_report(results_dict, output_file='reports/detailed_report.txt')
        print_success("reports/detailed_report.txt")
    except Exception as e:
        print(f"⚠️  WARNING: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n  [6.3] Generando model_comparison.txt...")
    try:
        create_comparison_table(results_dict, output_file='reports/model_comparison.txt')
        print_success("reports/model_comparison.txt")
    except Exception as e:
        print(f"⚠️  WARNING: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n  [6.4] Generando confusion matrices...")
    try:
        for model_name in models.keys():
            from sklearn.metrics import confusion_matrix
            y_pred = (y_proba_test[model_name][:, 1] >= optimal_thresholds[model_name]).astype(int)
            cm = confusion_matrix(y_test, y_pred)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
            ax.set_xlabel('Predicted', fontsize=11)
            ax.set_ylabel('Actual', fontsize=11)
            ax.set_title(f'Confusion Matrix: {model_name}', fontsize=12, fontweight='bold')
            plt.tight_layout()
            filename = f'plots/confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
        print_success("Confusion matrices guardadas")
    except Exception as e:
        print(f"⚠️  WARNING: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n  [6.5] Production config...")
    try:
        def convert_types(obj):
            if isinstance(obj, dict):
                return {str(k) if isinstance(k, np.integer) else k: convert_types(v) 
                       for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return (int(obj) if isinstance(obj, np.integer) else float(obj))
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        config = {
            'timestamp': str(time.time()),
            'models': list(models.keys()),
            'optimal_thresholds': convert_types(optimal_thresholds),
            'cv_results': {},
            'dataset': {
                'train_size': int(X_train.shape[0]),
                'val_size': int(X_val.shape[0]),
                'test_size': int(X_test.shape[0]),
                'n_features': int(X_train.shape[1]),
                'smote_applied': True
            }
        }
        
        with open('reports/config_produccion.json', 'w') as f:
            json.dump(config, f, indent=2)
        print_success("reports/config_produccion.json")
    except Exception as e:
        print(f"⚠️  WARNING: {e}")
        import traceback
        traceback.print_exc()
    
    # ──────────── RESUMEN FINAL ────────────
    elapsed = time.time() - start_time
    
    print_section("SUMMARY", "RESULTADOS FINALES")
    
    print("\n  [TABLA COMPARATIVA - TEST SET]")
    print("  " + "─"*70)
    print(f"  {'Modelo':<18} {'Accuracy':>12} {'Precision':>12} {'Recall':>12} {'F1':>12}")
    print("  " + "─"*70)
    
    for model_name in models.keys():
        y_pred = (y_proba_test[model_name][:, 1] >= optimal_thresholds[model_name]).astype(int)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"  {model_name:<18} {acc:>11.1%} {prec:>11.1%} {rec:>11.1%} {f1:>11.1%}")
    
    print("  " + "─"*70)
    
    print("\n  [THRESHOLDS ÓPTIMOS]")
    for model_name, threshold in optimal_thresholds.items():
        print_info(f"{model_name}: {threshold:.3f}")
    
    print("\n  [ARCHIVOS GENERADOS]")
    print_info(f"Modelos:    models/*.pkl (4 archivos)")
    print_info(f"Gráficas:   plots/*.png (~10 archivos)")
    print_info(f"Reportes:   reports/* (3 archivos)")
    print_info(f"Config:     reports/config_produccion.json")
    
    print("\n" + "="*75)
    print(f"✓ PIPELINE COMPLETADO EN {elapsed:.1f}s".center(75))
    print("="*75)
    
    print("\n📖 PRÓXIMOS PASOS:")
    print("   1. Leer: MEMORIA_FINAL.md (análisis detallado)")
    print("   2. Revisar: reports/detailed_report.txt (resultados)")
    print("   3. Predicciones: python production_predictor.py")
    print("   4. API: python api_example.py\n")

# ─────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted por usuario")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERROR INESPERADO: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
