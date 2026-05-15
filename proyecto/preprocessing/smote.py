"""
Módulo SMOTE: Técnica de sobremuestreo para balancear clases
SMOTE = Synthetic Minority Over-sampling Technique

Ventajas:
  - Genera muestras sintéticas de la clase minoritaria
  - Evita overfitting (a diferencia de simple duplication)
  - Mejora recall de clase minoritaria
  - ROC-AUC generalmente mejora
  
Documentación: https://imbalanced-learn.org/
"""

import numpy as np
from imblearn.over_sampling import SMOTE as ImbSMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek
from sklearn.utils.class_weight import compute_class_weight


def aplicar_smote(X_train, y_train, random_state=42, verbose=True):
    """
    Aplica SMOTE para balancear clases en datos de entrenamiento.
    
    Parámetros:
    -----------
    X_train : np.ndarray
        Datos de entrenamiento (n_samples, n_features)
    y_train : np.ndarray
        Etiquetas de entrenamiento
    random_state : int
        Seed para reproducibilidad
    verbose : bool
        Mostrar información del proceso
    
    Retorno:
    --------
    X_train_smote : np.ndarray
        Datos balanceados (más muestras)
    y_train_smote : np.ndarray
        Etiquetas balanceadas
    info : dict
        Información sobre el balanceo
    
    Ejemplo:
    --------
    X_train_smote, y_train_smote, info = aplicar_smote(X_train, y_train)
    print(f"Antes: {len(X_train)} muestras")
    print(f"Después: {len(X_train_smote)} muestras")
    print(f"Distribución: {info['distribucion_después']}")
    """
    
    if verbose:
        print("\n" + "="*70)
        print("APLICANDO SMOTE - Balanceo de Clases")
        print("="*70)
        
        # Contar clases antes
        unique, counts = np.unique(y_train, return_counts=True)
        print(f"\nANTES de SMOTE:")
        for label, count in zip(unique, counts):
            pct = 100 * count / len(y_train)
            print(f"  Clase {label}: {count:4d} muestras ({pct:5.1f}%)")
    
    # Aplicar SMOTE
    smote = ImbSMOTE(random_state=random_state, k_neighbors=5)
    
    try:
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    except Exception as e:
        print(f"⚠️ Error en SMOTE: {e}")
        print("   Retornando datos originales")
        return X_train, y_train, {'error': str(e)}
    
    # Información después
    if verbose:
        unique_after, counts_after = np.unique(y_train_smote, return_counts=True)
        print(f"\nDESPUÉS de SMOTE:")
        for label, count in zip(unique_after, counts_after):
            pct = 100 * count / len(y_train_smote)
            print(f"  Clase {label}: {count:4d} muestras ({pct:5.1f}%)")
        
        print(f"\n📊 ESTADÍSTICAS:")
        print(f"  Muestras originales: {len(X_train)}")
        print(f"  Muestras después SMOTE: {len(X_train_smote)}")
        print(f"  Aumento: +{len(X_train_smote) - len(X_train)} muestras (+{100*(len(X_train_smote)/len(X_train) - 1):.1f}%)")
        print(f"  Features mantenidos: {X_train_smote.shape[1]}")
        print("="*70 + "\n")
    
    info = {
        'muestras_antes': len(X_train),
        'muestras_después': len(X_train_smote),
        'aumento_pct': 100 * (len(X_train_smote) / len(X_train) - 1),
        'distribucion_antes': dict(zip(unique, counts)),
        'distribucion_después': dict(zip(unique_after, counts_after))
    }
    
    return X_train_smote, y_train_smote, info


def aplicar_smote_tomek(X_train, y_train, random_state=42, verbose=True):
    """
    Aplica SMOTE + Tomek Links para balanceo y limpieza de fronteras.
    
    Combina:
    - SMOTE: Sobremuestrea la clase minoritaria
    - Tomek Links: Elimina ejemplos conflictivos de la mayoría
    
    Resultado: Mejora tanto recall como precision
    """
    
    if verbose:
        print("\n" + "="*70)
        print("APLICANDO SMOTE + TOMEK LINKS")
        print("="*70)
        
        unique, counts = np.unique(y_train, return_counts=True)
        print(f"\nANTES:")
        for label, count in zip(unique, counts):
            pct = 100 * count / len(y_train)
            print(f"  Clase {label}: {count:4d} ({pct:5.1f}%)")
    
    # SMOTE + Tomek
    smt = ImbSMOTE(random_state=random_state)
    tomek = TomekLinks()
    
    try:
        X_resampled, y_resampled = smt.fit_resample(X_train, y_train)
        X_resampled, y_resampled = tomek.fit_resample(X_resampled, y_resampled)
    except Exception as e:
        print(f"⚠️ Error: {e}")
        return X_train, y_train, {'error': str(e)}
    
    if verbose:
        unique_after, counts_after = np.unique(y_resampled, return_counts=True)
        print(f"\nDESPUÉS:")
        for label, count in zip(unique_after, counts_after):
            pct = 100 * count / len(y_resampled)
            print(f"  Clase {label}: {count:4d} ({pct:5.1f}%)")
        print("="*70 + "\n")
    
    info = {
        'muestras_antes': len(X_train),
        'muestras_después': len(X_resampled),
        'aumento_pct': 100 * (len(X_resampled) / len(X_train) - 1),
        'distribucion_antes': dict(zip(unique, counts)),
        'distribucion_después': dict(zip(unique_after, counts_after))
    }
    
    return X_resampled, y_resampled, info


def comparar_con_sin_smote(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Comparación visual del impacto de SMOTE.
    
    Muestra:
    - Distribución de clases antes/después
    - Impacto esperado en recall
    """
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, f1_score
    
    print("\n" + "="*70)
    print("COMPARACIÓN: CON vs SIN SMOTE")
    print("="*70)
    
    # Sin SMOTE
    print("\n[1] MODELO SIN SMOTE (baseline)")
    print("-" * 70)
    
    rf_sin_smote = RandomForestClassifier(n_estimators=100, random_state=42, 
                                           class_weight='balanced', n_jobs=-1)
    rf_sin_smote.fit(X_train, y_train)
    y_pred_sin = rf_sin_smote.predict(X_test)
    f1_sin = f1_score(y_test, y_pred_sin, average='macro')
    
    print(classification_report(y_test, y_pred_sin, 
                               target_names=['Principiante', 'Intermedio']))
    print(f"F1-Score (macro): {f1_sin:.4f}")
    
    # Con SMOTE
    print("\n[2] MODELO CON SMOTE")
    print("-" * 70)
    
    X_train_smote, y_train_smote, info = aplicar_smote(X_train, y_train, verbose=False)
    
    rf_con_smote = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_con_smote.fit(X_train_smote, y_train_smote)
    y_pred_con = rf_con_smote.predict(X_test)
    f1_con = f1_score(y_test, y_pred_con, average='macro')
    
    print(classification_report(y_test, y_pred_con, 
                               target_names=['Principiante', 'Intermedio']))
    print(f"F1-Score (macro): {f1_con:.4f}")
    
    # Resumen
    print("\n" + "="*70)
    print("RESUMEN COMPARATIVO")
    print("="*70)
    print(f"F1-Score sin SMOTE:  {f1_sin:.4f}")
    print(f"F1-Score con SMOTE:  {f1_con:.4f}")
    print(f"Mejora:              +{(f1_con - f1_sin):.4f} ({100*(f1_con/f1_sin - 1):.1f}%)")
    print("="*70 + "\n")
    
    return {
        'sin_smote': {'f1': f1_sin, 'y_pred': y_pred_sin},
        'con_smote': {'f1': f1_con, 'y_pred': y_pred_con},
        'smote_info': info
    }
