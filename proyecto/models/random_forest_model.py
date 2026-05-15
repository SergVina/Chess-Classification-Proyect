"""
Módulo de Random Forest para clasificación multiclase.
Utiliza GridSearchCV de scikit-learn para optimización
de hiperparámetros e importancia de features.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def train_random_forest(X_train, y_train, X_val, y_val):
    """
    Entrena un modelo Random Forest usando GridSearchCV para optimización
    de hiperparámetros.
    
    Búsqueda de hiperparámetros:
        n_estimators ∈ {50, 100, 200}
        max_depth ∈ {None, 5, 10, 20}
        min_samples_split ∈ {2, 5, 10}
    
    Parámetros:
    -----------
    X_train : np.ndarray
        Features de entrenamiento
    y_train : np.ndarray
        Etiquetas de entrenamiento
    X_val : np.ndarray
        Features de validación
    y_val : np.ndarray
        Etiquetas de validación
    
    Retorno:
    --------
    dict : Diccionario con:
        - 'best_estimator': Mejor modelo RF encontrado
        - 'best_params': Mejores hiperparámetros
        - 'best_score': Mejor score en CV
        - 'feature_importance': Importancia de features
        - 'cv_results': Resultados completos de GridSearchCV
    """
    print("Entrenando Random Forest con GridSearchCV...")
    
    # Definir grid de hiperparámetros
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    
    # Crear base Random Forest con balanceo de clases
    rf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
    
    # Grid search con 5-fold cross-validation
    grid_search = GridSearchCV(
        rf,
        param_grid,
        cv=5,
        n_jobs=-1,
        verbose=1,
        scoring='accuracy'
    )
    
    # Entrenar
    grid_search.fit(X_train, y_train)
    
    print(f"\nMejores parámetros Random Forest: {grid_search.best_params_}")
    print(f"Mejor CV score: {grid_search.best_score_:.4f}")
    
    # Evaluar en validación
    best_estimator = grid_search.best_estimator_
    val_score = best_estimator.score(X_val, y_val)
    print(f"Scoring en validación: {val_score:.4f}")
    
    # Extraer importancia de features
    feature_importance = best_estimator.feature_importances_
    
    return {
        'best_estimator': best_estimator,
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'feature_importance': feature_importance,
        'cv_results': grid_search.cv_results_,
        'grid_search': grid_search
    }
