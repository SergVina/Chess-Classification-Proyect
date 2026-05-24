import numpy as np
import pandas as pd
import os
import pickle
from preprocessing.preprocess import preprocess_pipeline
from preprocessing.smote import aplicar_smote
from models.random_forest_model import train_random_forest
from models.svm_model import train_svm
from evaluation.metrics import evaluate_model
from sklearn.metrics import f1_score, recall_score, precision_score
from datetime import datetime


def main():
    print("="*100)
    print("COMPARACIÓN: IMPACTO DE SMOTE EN EL ENTRENAMIENTO")
    print("="*100)
    
    print("\n[1/3] Cargando y preprocesando datos...")
    csv_path = os.path.join('data', 'games.csv')
    
    preprocess_result = preprocess_pipeline(
        csv_path,
        n_samples=2500,
        plots_dir='plots',
        skip_eda=True
    )
    
    data_split = preprocess_result['data_split']
    X_train = data_split['X_train']
    X_val = data_split['X_val']
    X_test = data_split['X_test']
    y_train = data_split['y_train']
    y_val = data_split['y_val']
    y_test = data_split['y_test']
    
    print(f"✓ Datos cargados")
    print(f"  Train: {len(X_train)} muestras")
    print(f"  Val: {len(X_val)} muestras")
    print(f"  Test: {len(X_test)} muestras")
    
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"\nDistribución original en train:")
    for label, count in zip(unique, counts):
        pct = 100 * count / len(y_train)
        print(f"  Clase {label}: {count} ({pct:.1f}%)")
    
    print("\n[2/3] ENTRENAMIENTO SIN SMOTE")
    print("-" * 100)
    
    print("\nEntrenando Random Forest (baseline)...")
    rf_result_sin_smote = train_random_forest(data_split, X_train, y_train, X_val, y_val)
    rf_sin_smote = rf_result_sin_smote['best_estimator']
    
    metrics_sin_smote = evaluate_model(rf_sin_smote, X_test, y_test, model_name='Random Forest (SIN SMOTE)')
    
    print(f"\nResultados SIN SMOTE:")
    print(f"  Accuracy: {metrics_sin_smote['accuracy']:.4f}")
    print(f"  F1-Score (macro): {metrics_sin_smote['f1_macro']:.4f}")
    print(f"  Precision (macro): {metrics_sin_smote['precision_macro']:.4f}")
    print(f"  Recall (macro): {metrics_sin_smote['recall_macro']:.4f}")
    
    y_pred_sin = rf_sin_smote.predict(X_test)
    recall_por_clase_sin = recall_score(y_test, y_pred_sin, average=None)
    print(f"\n  Recall por clase:")
    for i, r in enumerate(recall_por_clase_sin):
        print(f"    Clase {i}: {r:.4f}")
    
    print("\n[3/3] ENTRENAMIENTO CON SMOTE")
    print("-" * 100)
    
    print("\nAplicando SMOTE...")
    X_train_smote, y_train_smote, smote_info = aplicar_smote(X_train, y_train, verbose=False)
    
    print(f"\nDistribución después de SMOTE:")
    unique_smote, counts_smote = np.unique(y_train_smote, return_counts=True)
    for label, count in zip(unique_smote, counts_smote):
        pct = 100 * count / len(y_train_smote)
        print(f"  Clase {label}: {count} ({pct:.1f}%)")
    print(f"\nMuestras: {len(X_train)} → {len(X_train_smote)} (+{100*(len(X_train_smote)/len(X_train)-1):.1f}%)")
    
    print("\nEntrenando Random Forest (con SMOTE data)...")
    
    from models.random_forest_model import train_random_forest
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    
    params = {
        'n_estimators': [150],
        'max_depth': [15],
        'min_samples_split': [2]
    }
    
    rf_model = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
    grid_search = GridSearchCV(rf_model, params, cv=5, scoring='f1_macro', n_jobs=-1)
    grid_search.fit(X_train_smote, y_train_smote)
    
    rf_con_smote = grid_search.best_estimator_
    
    metrics_con_smote = evaluate_model(rf_con_smote, X_test, y_test, model_name='Random Forest (CON SMOTE)')
    
    print(f"\nResultados CON SMOTE:")
    print(f"  Accuracy: {metrics_con_smote['accuracy']:.4f}")
    print(f"  F1-Score (macro): {metrics_con_smote['f1_macro']:.4f}")
    print(f"  Precision (macro): {metrics_con_smote['precision_macro']:.4f}")
    print(f"  Recall (macro): {metrics_con_smote['recall_macro']:.4f}")
    
    y_pred_con = rf_con_smote.predict(X_test)
    recall_por_clase_con = recall_score(y_test, y_pred_con, average=None)
    print(f"\n  Recall por clase:")
    for i, r in enumerate(recall_por_clase_con):
        print(f"    Clase {i}: {r:.4f}")
    
    print("\n" + "="*100)
    print("COMPARACIÓN: ANTES vs DESPUÉS de SMOTE")
    print("="*100)
    
    print(f"\n{'Métrica':<25} | {'SIN SMOTE':<15} | {'CON SMOTE':<15} | {'Mejora':<15}")
    print("-" * 75)
    
    metrics_para_comparar = [
        ('Accuracy', metrics_sin_smote['accuracy'], metrics_con_smote['accuracy']),
        ('F1-Score (macro)', metrics_sin_smote['f1_macro'], metrics_con_smote['f1_macro']),
        ('Precision (macro)', metrics_sin_smote['precision_macro'], metrics_con_smote['precision_macro']),
        ('Recall (macro)', metrics_sin_smote['recall_macro'], metrics_con_smote['recall_macro']),
        ('Recall Clase 0', recall_por_clase_sin[0], recall_por_clase_con[0]),
        ('Recall Clase 1', recall_por_clase_sin[1], recall_por_clase_con[1]),
    ]
    
    mejoras = []
    for nombre, sin, con in metrics_para_comparar:
        diff = con - sin
        pct = 100 * (con / sin - 1) if sin != 0 else 0
        mejoras.append(pct)
        
        marca = "📈" if con > sin else "📉" if con < sin else "="
        print(f"{nombre:<25} | {sin:>14.4f} | {con:>14.4f} | {marca} {pct:+7.2f}%")
    
    print("\n" + "="*100)
    print("CONCLUSIÓN")
    print("="*100)
    
    mejora_promedio = np.mean(mejoras)
    print(f"\nMejora promedio: {mejora_promedio:+.2f}%")
    
    if mejora_promedio > 0:
        print(f"\n✅ SMOTE MEJORÓ el modelo en {mejora_promedio:.2f}% en promedio")
        print("\n📊 Recomendación: Usar SMOTE en producción")
    else:
        print(f"\n⚠️ SMOTE no mejoró significativamente (cambio: {mejora_promedio:+.2f}%)")
        print("\n📊 Recomendación: Verificar configuración")
    
    print(f"\n📝 Detalles:")
    print(f"  Muestras train sin SMOTE: {len(X_train)}")
    print(f"  Muestras train con SMOTE: {len(X_train_smote)}")
    print(f"  Aumento de datos: +{100*(len(X_train_smote)/len(X_train)-1):.1f}%")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n" + "="*100 + "\n")


if __name__ == '__main__':
    main()
