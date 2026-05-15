"""
Módulo de Máquina de Vectores de Soporte (SVM) con kernel RBF.
Utiliza GridSearchCV de scikit-learn para optimizar hiperparámetros.
"""

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


def train_svm(X_train, y_train, X_val, y_val):
    """
    Entrena un modelo SVM con kernel RBF usando GridSearchCV para optimización
    de hiperparámetros.
    
    Búsqueda de hiperparámetros:
        C ∈ {0.1, 1, 10, 100}
        gamma ∈ {'scale', 'auto', 0.01, 0.1}
    
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
        - 'best_estimator': Mejor modelo SVM encontrado
        - 'best_params': Mejores hiperparámetros
        - 'best_score': Mejor score en CV
        - 'cv_results': Resultados completos de GridSearchCV
    """
    print("Entrenando SVM con GridSearchCV...")
    
    # Definir grid de hiperparámetros
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.1]
    }
    
    # Crear base SVM con kernel RBF y balanceo de clases
    svm = SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced')
    
    # Grid search con 5-fold cross-validation
    grid_search = GridSearchCV(
        svm,
        param_grid,
        cv=5,
        n_jobs=-1,
        verbose=1,
        scoring='accuracy'
    )
    
    # Entrenar
    grid_search.fit(X_train, y_train)
    
    print(f"\nMejores parámetros SVM: {grid_search.best_params_}")
    print(f"Mejor CV score: {grid_search.best_score_:.4f}")
    
    # Evaluar en validación
    best_estimator = grid_search.best_estimator_
    val_score = best_estimator.score(X_val, y_val)
    print(f"Scoring en validación: {val_score:.4f}")
    
    return {
        'best_estimator': best_estimator,
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'cv_results': grid_search.cv_results_,
        'grid_search': grid_search
    }
