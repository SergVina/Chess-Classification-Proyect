"""
Módulo de evaluación y visualización de modelos.
Contiene funciones para:
  - Calcular métricas (accuracy, precision, recall, F1)
  - Visualizar matrices de confusión
  - Generar curvas de aprendizaje
  - Validación cruzada estratificada
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend non-interactive (sin tkinter warnings)
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
)
from sklearn.model_selection import learning_curve, StratifiedKFold, cross_val_score


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evalúa un modelo en el conjunto de test.
    
    Parámetros:
    -----------
    model : objeto
        Modelo entrenado con métodos predict()
    X_test : np.ndarray
        Features de test
    y_test : np.ndarray
        Etiquetas de test
    model_name : str
        Nombre del modelo para display
    
    Retorno:
    --------
    dict : Diccionario con todas las métricas
    """
    y_pred = model.predict(X_test)
    
    # Calcular número de clases
    n_classes = len(np.unique(y_test))
    class_names_all = ['Principiante', 'Intermedio', 'Avanzado']
    class_names = class_names_all[:n_classes]
    
    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    # Precision, recall, f1 por clase
    precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)
    
    # Classification report
    class_report = classification_report(
        y_test, y_pred,
        target_names=class_names,
        zero_division=0,
        labels=np.arange(n_classes)
    )
    
    # Matriz de confusión
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Calcular ROC-AUC (para clasificación binaria)
    roc_auc = None
    y_proba = None
    if n_classes == 2 and hasattr(model, 'predict_proba'):
        try:
            y_proba = model.predict_proba(X_test)
            roc_auc = roc_auc_score(y_test, y_proba[:, 1])
        except:
            pass
    
    print(f"\n{'='*60}")
    print(f"EVALUACIÓN: {model_name}")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (macro): {precision_macro:.4f}")
    print(f"Recall (macro): {recall_macro:.4f}")
    print(f"F1-Score (macro): {f1_macro:.4f}")
    if roc_auc is not None:
        print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"\n{class_report}")
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'roc_auc': roc_auc,
        'y_proba': y_proba,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'conf_matrix': conf_matrix,
        'y_pred': y_pred,
        'class_report': class_report
    }


def plot_confusion_matrix(conf_matrix, model_name, plots_dir='plots'):
    """
    Visualiza y guarda la matriz de confusión.
    
    Parámetros:
    -----------
    conf_matrix : np.ndarray
        Matriz de confusión (n_classes x n_classes)
    model_name : str
        Nombre del modelo
    plots_dir : str
        Directorio para guardar
    """
    os.makedirs(plots_dir, exist_ok=True)
    
    class_names_all = ['Principiante', 'Intermedio', 'Avanzado']
    n_classes = conf_matrix.shape[0]
    class_names = class_names_all[:n_classes]
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Visualizar matriz
    im = ax.imshow(conf_matrix, cmap='Blues', aspect='auto')
    
    # Configurar ejes
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    
    # Añadir valores en las células
    for i in range(n_classes):
        for j in range(n_classes):
            text = ax.text(j, i, conf_matrix[i, j],
                          ha='center', va='center',
                          color='white' if conf_matrix[i, j] > conf_matrix.max() / 2 else 'black',
                          fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Predicción', fontsize=12, fontweight='bold')
    ax.set_ylabel('Real', fontsize=12, fontweight='bold')
    ax.set_title(f'Matriz de Confusión - {model_name}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Sanitizar nombre para guardar
    safe_name = model_name.replace('/', '_').replace(' ', '_')
    plt.savefig(os.path.join(plots_dir, f'confusion_matrix_{safe_name}.png'), dpi=150)
    plt.close()
    
    print(f"  ✓ Matriz de confusión guardada: confusion_matrix_{safe_name}.png")


def plot_learning_curves(model, X_train, y_train, model_name, plots_dir='plots'):
    """
    Genera y guarda curvas de aprendizaje (train score vs CV score).
    
    Parámetros:
    -----------
    model : objeto
        Modelo con método predict()
    X_train : np.ndarray
        Features de entrenamiento
    y_train : np.ndarray
        Etiquetas de entrenamiento
    model_name : str
        Nombre del modelo
    plots_dir : str
        Directorio para guardar
    """
    os.makedirs(plots_dir, exist_ok=True)
    
    print(f"  Generando curvas de aprendizaje para {model_name}...")
    
    train_sizes, train_scores, val_scores = learning_curve(
        model,
        X_train, y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1,
        train_sizes=np.linspace(0.2, 1.0, 9),
        scoring='accuracy',
        error_score=np.nan,
    )
    
    # Calcular media y desviación estándar (ignorando NaN de folds fallidos)
    train_mean = np.nanmean(train_scores, axis=1)
    train_std = np.nanstd(train_scores, axis=1)
    val_mean = np.nanmean(val_scores, axis=1)
    val_std = np.nanstd(val_scores, axis=1)
    
    # Plotear
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(train_sizes, train_mean, 'o-', color='#2E86AB', linewidth=2, markersize=6, label='Training score')
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color='#2E86AB')
    
    ax.plot(train_sizes, val_mean, 'o-', color='#A23B72', linewidth=2, markersize=6, label='Cross-validation score')
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2, color='#A23B72')
    
    ax.set_xlabel('Tamaño del Conjunto de Entrenamiento', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(f'Curva de Aprendizaje - {model_name}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    
    safe_name = model_name.replace('/', '_').replace(' ', '_')
    plt.savefig(os.path.join(plots_dir, f'learning_curve_{safe_name}.png'), dpi=150)
    plt.close()
    
    print(f"  ✓ Curva de aprendizaje guardada: learning_curve_{safe_name}.png")


def plot_nn_training_history(history, best_epoch, plots_dir='plots'):
    """
    Visualiza historial de entrenamiento de la red neuronal.
    
    Parámetros:
    -----------
    history : dict
        Diccionario con 'train_loss', 'val_loss', 'train_acc', 'val_acc'
    best_epoch : int
        Época con mejor validación para marcar early stopping
    plots_dir : str
        Directorio para guardar
    """
    os.makedirs(plots_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = np.arange(len(history['train_loss']))
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], 'o-', color='#2E86AB', linewidth=2, markersize=5, label='Training Loss')
    axes[0].plot(epochs, history['val_loss'], 'o-', color='#A23B72', linewidth=2, markersize=5, label='Validation Loss')
    axes[0].axvline(best_epoch, color='green', linestyle='--', linewidth=2, label=f'Early Stop (Época {best_epoch+1})')
    axes[0].set_xlabel('Época', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Loss (Cross-Entropy)', fontsize=12, fontweight='bold')
    axes[0].set_title('Pérdida de Entrenamiento vs Validación', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(epochs, history['train_acc'], 'o-', color='#2E86AB', linewidth=2, markersize=5, label='Training Accuracy')
    axes[1].plot(epochs, history['val_acc'], 'o-', color='#A23B72', linewidth=2, markersize=5, label='Validation Accuracy')
    axes[1].axvline(best_epoch, color='green', linestyle='--', linewidth=2, label=f'Early Stop (Época {best_epoch+1})')
    axes[1].set_xlabel('Época', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    axes[1].set_title('Precisión de Entrenamiento vs Validación', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'nn_training_history.png'), dpi=150)
    plt.close()
    
    print(f"  ✓ Historial de entrenamiento NN guardado: nn_training_history.png")
    
    # Análisis
    print(f"\n  ANÁLISIS NN:")
    final_train_loss = history['train_loss'][-1]
    final_val_loss = history['val_loss'][-1]
    final_train_acc = history['train_acc'][-1]
    final_val_acc = history['val_acc'][-1]
    
    if final_val_loss < final_train_loss:
        print(f"    → Validación mejor que entrenamiento: UNDERFITTING potencial")
    elif final_train_loss < final_val_loss - 0.05:
        print(f"    → Validación peor que entrenamiento: OVERFITTING detectado")
    else:
        print(f"    → Buen ajuste (train y validation cercanos)")


def stratified_cv_evaluation(model, X, y, cv=5):
    """
    Realiza validación cruzada estratificada 5-fold (solo para modelos sklearn).
    Para modelos personalizados, retorna None.
    
    Parámetros:
    -----------
    model : objeto
        Modelo con método score()
    X : np.ndarray
        Features
    y : np.ndarray
        Etiquetas
    cv : int
        Número de folds
    
    Retorno:
    --------
    dict : Media ± desviación estándar de accuracy, o None si no es modelo sklearn
    """
    # Verificar si es modelo sklearn (tiene __sklearn_tags__)
    try:
        if not hasattr(model, '__sklearn_tags__'):
            # Modelo personalizado, no hacer CV
            return None
        
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy', n_jobs=-1)
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        return {
            'cv_scores': scores,
            'mean': mean_score,
            'std': std_score
        }
    except Exception as e:
        # Si falla, retornar None
        return None


def manual_cv_evaluation(model_class, model_kwargs, X, y, cv=5, is_neural_network=False):
    """
    Validación cruzada estratificada manual para modelos implementados con NumPy
    (LogisticRegressionOvA y NeuralNetwork) que no son compatibles con cross_val_score.

    Para NeuralNetwork, cada fold divide el train en train+val interno (80/20) para
    satisfacer la firma fit(X_train, y_train, X_val, y_val) con early stopping.

    Parámetros:
    -----------
    model_class : class
        Clase del modelo (LogisticRegressionOvA o NeuralNetwork)
    model_kwargs : dict
        Argumentos para instanciar el modelo
    X : np.ndarray
        Features completas (train+val)
    y : np.ndarray
        Etiquetas completas (train+val)
    cv : int
        Número de folds
    is_neural_network : bool
        True si el modelo es NeuralNetwork (requiere X_val en fit)

    Retorno:
    --------
    dict : {'cv_scores': np.ndarray, 'mean': float, 'std': float}
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_fold_train, X_fold_test = X[train_idx], X[test_idx]
        y_fold_train, y_fold_test = y[train_idx], y[test_idx]

        model = model_class(**model_kwargs)

        if is_neural_network:
            # Reservar 20% del train del fold como validación interna para early stopping
            val_split = int(0.2 * len(X_fold_train))
            X_fold_val = X_fold_train[:val_split]
            y_fold_val = y_fold_train[:val_split]
            X_fold_train_inner = X_fold_train[val_split:]
            y_fold_train_inner = y_fold_train[val_split:]
            model.fit(X_fold_train_inner, y_fold_train_inner, X_fold_val, y_fold_val)
        else:
            model.fit(X_fold_train, y_fold_train)

        fold_score = model.score(X_fold_test, y_fold_test)
        scores.append(fold_score)

    scores = np.array(scores)
    return {
        'cv_scores': scores,
        'mean': float(np.mean(scores)),
        'std': float(np.std(scores))
    }


# =============================================================================
# MEJORAS DIFERENCIALES
# =============================================================================

class EnsembleVoting:
    """
    Ensemble por promedio de probabilidades (soft voting) sobre los 4 modelos.
    Pesos opcionales por modelo según su ROC-AUC en validación.
    """

    def __init__(self, models, weights=None):
        """
        Parámetros:
        -----------
        models : dict  {nombre: modelo_entrenado}
        weights : dict {nombre: float} o None (pesos iguales)
        """
        self.models = models
        self.weights = weights

    def predict_proba(self, X):
        probas = []
        w_total = 0.0
        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                p = model.predict_proba(X)
                w = self.weights[name] if self.weights else 1.0
                probas.append(p * w)
                w_total += w
        return np.sum(probas, axis=0) / w_total

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= threshold).astype(int)

    def score(self, X, y):
        return float(np.mean(self.predict(X) == y))


def evaluate_ensemble(ensemble, X_test, y_test, plots_dir='plots'):
    """
    Evalúa el ensemble con las mismas métricas que el resto de modelos
    y guarda la matriz de confusión y curva ROC.

    Retorno:
    --------
    dict con las métricas (misma estructura que evaluate_model)
    """
    y_pred = ensemble.predict(X_test)
    y_proba = ensemble.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba[:, 1])
    conf_matrix = confusion_matrix(y_test, y_pred)

    n_classes = len(np.unique(y_test))
    class_names = ['Principiante', 'Intermedio', 'Avanzado'][:n_classes]
    class_report = classification_report(y_test, y_pred, target_names=class_names,
                                         zero_division=0, labels=np.arange(n_classes))

    print(f"\n{'='*60}")
    print("EVALUACIÓN: Ensemble (Soft Voting)")
    print(f"{'='*60}")
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Precision (macro): {precision_macro:.4f}")
    print(f"Recall (macro):    {recall_macro:.4f}")
    print(f"F1-Score (macro):  {f1_macro:.4f}")
    print(f"ROC-AUC:           {roc_auc:.4f}")
    print(f"\n{class_report}")

    plot_confusion_matrix(conf_matrix, 'Ensemble', plots_dir=plots_dir)
    plot_roc_curve(y_test, y_proba, 'Ensemble', plots_dir=plots_dir)

    return {
        'model_name': 'Ensemble',
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'roc_auc': roc_auc,
        'y_proba': y_proba,
        'precision_per_class': precision_score(y_test, y_pred, average=None, zero_division=0),
        'recall_per_class': recall_score(y_test, y_pred, average=None, zero_division=0),
        'f1_per_class': f1_score(y_test, y_pred, average=None, zero_division=0),
        'conf_matrix': conf_matrix,
        'y_pred': y_pred,
        'class_report': class_report,
    }


def find_optimal_threshold_for_recall(y_test, y_proba, min_precision=0.40):
    """
    Encuentra el threshold que maximiza el recall de Intermedios (clase 1)
    sujeto a precision mínima aceptable.

    Retorno:
    --------
    dict con threshold, f1, precision y recall de la clase Intermedio
    """
    best = {'threshold': 0.5, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
    for threshold in np.linspace(0.10, 0.70, 121):
        y_pred = (y_proba[:, 1] >= threshold).astype(int)
        prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        rec  = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1   = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        if prec >= min_precision and f1 > best['f1']:
            best = {'threshold': float(threshold), 'f1': float(f1),
                    'precision': float(prec), 'recall': float(rec)}
    return best


def evaluate_with_threshold(model_name, y_test, y_proba, threshold, plots_dir='plots'):
    """
    Re-evalúa un modelo usando un threshold ajustado y guarda la matriz de confusión.

    Retorno:
    --------
    dict con métricas completas bajo el threshold dado
    """
    y_pred = (y_proba[:, 1] >= threshold).astype(int)

    accuracy        = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall_macro    = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1_macro        = f1_score(y_test, y_pred, average='macro', zero_division=0)
    roc_auc         = roc_auc_score(y_test, y_proba[:, 1])
    conf_matrix     = confusion_matrix(y_test, y_pred)

    n_classes   = len(np.unique(y_test))
    class_names = ['Principiante', 'Intermedio', 'Avanzado'][:n_classes]
    class_report = classification_report(y_test, y_pred, target_names=class_names,
                                         zero_division=0, labels=np.arange(n_classes))

    label = f"{model_name} (thr={threshold:.3f})"
    print(f"\n{'='*60}")
    print(f"EVALUACIÓN CON THRESHOLD AJUSTADO: {label}")
    print(f"{'='*60}")
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Precision (macro): {precision_macro:.4f}")
    print(f"Recall (macro):    {recall_macro:.4f}")
    print(f"F1-Score (macro):  {f1_macro:.4f}")
    print(f"ROC-AUC:           {roc_auc:.4f}")
    print(f"\n{class_report}")

    safe = f"{model_name}_threshold".replace(' ', '_').replace('/', '_')
    plot_confusion_matrix(conf_matrix, label, plots_dir=plots_dir)

    return {
        'model_name': label,
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'roc_auc': roc_auc,
        'y_proba': y_proba,
        'precision_per_class': precision_score(y_test, y_pred, average=None, zero_division=0),
        'recall_per_class':    recall_score(y_test, y_pred, average=None, zero_division=0),
        'f1_per_class':        f1_score(y_test, y_pred, average=None, zero_division=0),
        'conf_matrix': conf_matrix,
        'y_pred': y_pred,
        'class_report': class_report,
        'threshold': threshold,
    }


def plot_validation_curve(model_class, model_kwargs, param_name, param_range,
                          X, y, plots_dir='plots', model_name='Model',
                          is_neural_network=False):
    """
    Genera la curva de validación de un hiperparámetro para modelos NumPy.
    Muestra train score y CV score (5-fold) en función del valor del parámetro.

    Parámetros:
    -----------
    model_class : class
    model_kwargs : dict  (kwargs base; param_name se sobreescribe en cada iteración)
    param_name : str     nombre del hiperparámetro a variar
    param_range : array  valores a explorar
    X, y : arrays
    is_neural_network : bool
    """
    os.makedirs(plots_dir, exist_ok=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_means, train_stds = [], []
    val_means, val_stds = [], []

    for param_val in param_range:
        kwargs = {**model_kwargs, param_name: param_val}
        fold_train_scores, fold_val_scores = [], []

        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_vl = X[train_idx], X[val_idx]
            y_tr, y_vl = y[train_idx], y[val_idx]

            model = model_class(**kwargs)

            if is_neural_network:
                split = int(0.2 * len(X_tr))
                model.fit(X_tr[split:], y_tr[split:], X_tr[:split], y_tr[:split])
            else:
                model.fit(X_tr, y_tr)

            fold_train_scores.append(model.score(X_tr, y_tr))
            fold_val_scores.append(model.score(X_vl, y_vl))

        train_means.append(np.mean(fold_train_scores))
        train_stds.append(np.std(fold_train_scores))
        val_means.append(np.mean(fold_val_scores))
        val_stds.append(np.std(fold_val_scores))

    train_means = np.array(train_means)
    train_stds = np.array(train_stds)
    val_means = np.array(val_means)
    val_stds = np.array(val_stds)

    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(param_range))

    ax.plot(x_pos, train_means, 'o-', color='#2E86AB', linewidth=2, markersize=6, label='Train score')
    ax.fill_between(x_pos, train_means - train_stds, train_means + train_stds, alpha=0.2, color='#2E86AB')

    ax.plot(x_pos, val_means, 'o-', color='#A23B72', linewidth=2, markersize=6, label='CV score (5-fold)')
    ax.fill_between(x_pos, val_means - val_stds, val_means + val_stds, alpha=0.2, color='#A23B72')

    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(v) for v in param_range], rotation=30, ha='right')
    ax.set_xlabel(param_name, fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(f'Curva de Validación — {model_name} ({param_name})', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    safe = f"{model_name}_{param_name}".replace(' ', '_').replace('/', '_')
    path = os.path.join(plots_dir, f'validation_curve_{safe}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ✓ Curva de validación guardada: validation_curve_{safe}.png")


def plot_lr_coefficients(lr_model, feature_names, plots_dir='plots'):
    """
    Visualiza los coeficientes de la Regresión Logística como proxy de importancia
    de features. Para OvA con 2 clases usa los pesos del clasificador de clase 1
    (Intermedio vs resto), que es el más informativo para el desbalance.

    Parámetros:
    -----------
    lr_model : LogisticRegressionOvA  entrenado
    feature_names : list[str]
    plots_dir : str
    """
    os.makedirs(plots_dir, exist_ok=True)

    if lr_model.weights is None or len(lr_model.weights) == 0:
        print("  ⚠ LR no entrenado, saltando coeficientes.")
        return

    # Usar el clasificador de clase 1 (Intermedio) para el gráfico principal
    coefs = lr_model.weights[1] if len(lr_model.weights) > 1 else lr_model.weights[0]

    if len(feature_names) != len(coefs):
        feature_names = [f'Feature {i}' for i in range(len(coefs))]

    sorted_idx = np.argsort(np.abs(coefs))[::-1]
    top_n = min(15, len(coefs))
    idx = sorted_idx[:top_n]

    colors = ['#E74C3C' if coefs[i] > 0 else '#3498DB' for i in idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(top_n), coefs[idx], color=colors, edgecolor='black', linewidth=0.8)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in idx])
    ax.invert_yaxis()
    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlabel('Coeficiente (↑ favorece Intermedio, ↓ favorece Principiante)',
                  fontsize=11, fontweight='bold')
    ax.set_title('Importancia de Features — Regresión Logística (clase Intermedio)',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#E74C3C', label='Favorece Intermedio'),
                       Patch(facecolor='#3498DB', label='Favorece Principiante')]
    ax.legend(handles=legend_elements, fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'lr_feature_coefficients.png'), dpi=150)
    plt.close()
    print("  ✓ Coeficientes LR guardados: lr_feature_coefficients.png")


def plot_feature_importance(feature_importance, feature_names=None, n_features_to_show=15, plots_dir='plots', model_name='Random Forest'):
    """
    Visualiza la importancia de features para Random Forest.
    
    Parámetros:
    -----------
    feature_importance : np.ndarray
        Array de importancia de features
    feature_names : list[str] | None
        Nombres reales de las features en el mismo orden que `feature_importance`
    n_features_to_show : int
        Número de top features a mostrar
    plots_dir : str
        Directorio para guardar
    model_name : str
        Nombre del modelo
    """
    os.makedirs(plots_dir, exist_ok=True)
    
    # Obtener top N features
    sorted_idx = np.argsort(feature_importance)[::-1][:n_features_to_show]
    top_importance = feature_importance[sorted_idx]
    if feature_names is not None and len(feature_names) == len(feature_importance):
        top_labels = [feature_names[i] for i in sorted_idx]
    else:
        top_labels = [f'Feature {i}' for i in sorted_idx]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.barh(range(len(top_importance)), top_importance, color='#2E86AB', edgecolor='black', linewidth=1)
    ax.set_yticks(range(len(top_importance)))
    ax.set_yticklabels(top_labels)
    ax.set_xlabel('Importancia Relativa', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {n_features_to_show} Features - {model_name}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()
    
    plt.tight_layout()
    safe_name = model_name.replace('/', '_').replace(' ', '_')
    plt.savefig(os.path.join(plots_dir, f'feature_importance_{safe_name}.png'), dpi=150)
    plt.close()
    
    print(f"  ✓ Importancia de features guardada: feature_importance_{safe_name}.png")


def plot_roc_curve(y_test, y_proba, model_name, plots_dir='plots'):
    """
    Visualiza la curva ROC para clasificación binaria.
    
    Parámetros:
    -----------
    y_test : np.ndarray
        Etiquetas verdaderas
    y_proba : np.ndarray
        Probabilidades predichas (n_samples, 2)
    model_name : str
        Nombre del modelo
    plots_dir : str
        Directorio para guardar
    """
    if y_proba is None or len(np.unique(y_test)) != 2:
        return
    
    os.makedirs(plots_dir, exist_ok=True)
    
    # Calcular curva ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    
    # Plotear
    fig, ax = plt.subplots(figsize=(8, 7))
    
    ax.plot(fpr, tpr, color='#2E86AB', linewidth=2.5, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=2, label='Random Classifier')
    
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title(f'Curva ROC - {model_name}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    
    plt.tight_layout()
    safe_name = model_name.replace('/', '_').replace(' ', '_')
    plt.savefig(os.path.join(plots_dir, f'roc_curve_{safe_name}.png'), dpi=150)
    plt.close()
    
    print(f"  ✓ Curva ROC guardada: roc_curve_{safe_name}.png (AUC: {roc_auc:.4f})")


def find_optimal_threshold(y_test, y_proba):
    """
    Encuentra el threshold óptimo para maximizar F1-score en clasificación binaria.
    
    Parámetros:
    -----------
    y_test : np.ndarray
        Etiquetas verdaderas
    y_proba : np.ndarray
        Probabilidades predichas (n_samples, 2)
    
    Retorno:
    --------
    dict : {'threshold': float, 'f1': float, 'precision': float, 'recall': float}
    """
    if y_proba is None or len(np.unique(y_test)) != 2:
        return None
    
    best_f1 = 0
    best_threshold = 0.5
    best_precision = 0
    best_recall = 0
    
    # Buscar threshold óptimo
    for threshold in np.linspace(0, 1, 100):
        y_pred_thresh = (y_proba[:, 1] >= threshold).astype(int)
        
        f1 = f1_score(y_test, y_pred_thresh, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_precision = precision_score(y_test, y_pred_thresh, zero_division=0)
            best_recall = recall_score(y_test, y_pred_thresh, zero_division=0)
    
    return {
        'threshold': best_threshold,
        'f1': best_f1,
        'precision': best_precision,
        'recall': best_recall
    }


def generate_detailed_report(results_dict, output_file='detailed_report.txt'):
    """
    Genera un reporte detallado de todos los modelos.
    
    Parámetros:
    -----------
    results_dict : dict
        Diccionario con resultados {model_name: metrics_dict}
    output_file : str
        Archivo de salida
    """
    with open(output_file, 'w') as f:
        f.write("="*100 + "\n")
        f.write("REPORTE DETALLADO DE EVALUACIÓN DE MODELOS\n")
        f.write("="*100 + "\n\n")
        
        for model_name, metrics in results_dict.items():
            f.write(f"{'='*60}\n")
            f.write(f"MODELO: {model_name}\n")
            f.write(f"{'='*60}\n\n")
            
            f.write("MÉTRICAS GENERALES:\n")
            f.write(f"  Accuracy:      {metrics['accuracy']:.4f}\n")
            f.write(f"  Precision (macro): {metrics['precision_macro']:.4f}\n")
            f.write(f"  Recall (macro):    {metrics['recall_macro']:.4f}\n")
            f.write(f"  F1-Score (macro):  {metrics['f1_macro']:.4f}\n")
            
            if metrics.get('roc_auc') is not None:
                f.write(f"  ROC-AUC:       {metrics['roc_auc']:.4f}\n")
            
            f.write(f"\nCLASIFICATION REPORT:\n")
            f.write(f"{metrics['class_report']}\n")
            
            f.write(f"\nMATRIZ DE CONFUSIÓN:\n")
            conf_matrix = metrics['conf_matrix']
            f.write(f"{conf_matrix}\n\n")
            
            f.write("\n")
    
    print(f"Reporte detallado guardado en {output_file}")


def create_comparison_table(results_dict, output_file='model_comparison.txt'):
    """
    Crea una tabla comparativa de todos los modelos.
    
    Parámetros:
    -----------
    results_dict : dict
        Diccionario con resultados {model_name: metrics_dict}
    output_file : str
        Archivo de salida
    """
    print(f"\n{'='*120}")
    print("TABLA COMPARATIVA DE MODELOS")
    print(f"{'='*120}")
    
    # Determinar si hay ROC-AUC disponible
    has_roc = any(metrics.get('roc_auc') is not None for metrics in results_dict.values())
    
    if has_roc:
        header = f"{'Modelo':<25} | {'Accuracy':<12} | {'Precision':<12} | {'Recall':<12} | {'F1-Score':<12} | {'ROC-AUC':<10}"
        print(header)
        print("-" * 120)
        
        for model_name, metrics in results_dict.items():
            acc = metrics['accuracy']
            prec = metrics['precision_macro']
            rec = metrics['recall_macro']
            f1 = metrics['f1_macro']
            roc = metrics.get('roc_auc')
            roc_str = f"{roc:.4f}" if roc is not None else "N/A"
            
            row = f"{model_name:<25} | {acc:<12.4f} | {prec:<12.4f} | {rec:<12.4f} | {f1:<12.4f} | {roc_str:<10}"
            print(row)
    else:
        header = f"{'Modelo':<25} | {'Accuracy':<12} | {'Precision':<12} | {'Recall':<12} | {'F1-Score':<12}"
        print(header)
        print("-" * 100)
        
        for model_name, metrics in results_dict.items():
            acc = metrics['accuracy']
            prec = metrics['precision_macro']
            rec = metrics['recall_macro']
            f1 = metrics['f1_macro']
            
            row = f"{model_name:<25} | {acc:<12.4f} | {prec:<12.4f} | {rec:<12.4f} | {f1:<12.4f}"
            print(row)
    
    print(f"{'='*120}\n")
    
    # Guardar en archivo
    with open(output_file, 'w') as f:
        f.write("="*120 + "\n")
        f.write("TABLA COMPARATIVA DE MODELOS\n")
        f.write("="*120 + "\n")
        f.write(header + "\n")
        f.write("-" * (120 if has_roc else 100) + "\n")
        
        for model_name, metrics in results_dict.items():
            acc = metrics['accuracy']
            prec = metrics['precision_macro']
            rec = metrics['recall_macro']
            f1 = metrics['f1_macro']
            
            if has_roc:
                roc = metrics.get('roc_auc')
                roc_str = f"{roc:.4f}" if roc is not None else "N/A"
                row = f"{model_name:<25} | {acc:<12.4f} | {prec:<12.4f} | {rec:<12.4f} | {f1:<12.4f} | {roc_str:<10}"
            else:
                row = f"{model_name:<25} | {acc:<12.4f} | {prec:<12.4f} | {rec:<12.4f} | {f1:<12.4f}"
            f.write(row + "\n")
        
        f.write("="*120 + "\n")
    
    print(f"Tabla comparativa guardada en {output_file}")


def plot_models_comparison(results_dict, plots_dir='plots', metrics_to_show=None):
    """
    Crea una imagen única que compara varios modelos en métricas seleccionadas.

    Parámetros:
    -----------
    results_dict : dict
        Diccionario {model_name: metrics_dict}
    plots_dir : str
        Directorio donde guardar la imagen
    metrics_to_show : list[str] | None
        Lista de métricas a mostrar (por defecto: ['accuracy','f1_macro','roc_auc'])
    """
    if metrics_to_show is None:
        metrics_to_show = ['accuracy', 'f1_macro', 'roc_auc']

    os.makedirs(plots_dir, exist_ok=True)

    # Excluir comparaciones con threshold ajustado que se generan después
    def _is_threshold_variant(name):
        return '(thr=' in name or 'threshold' in name or 'thr=' in name

    filtered = {k: v for k, v in results_dict.items() if not _is_threshold_variant(k)}
    if len(filtered) == 0:
        filtered = results_dict

    model_names = list(filtered.keys())
    n_models = len(model_names)

    # Recolectar valores (NaN cuando falte una métrica)
    data = {}
    for metric in metrics_to_show:
        vals = []
        for m in model_names:
            v = filtered[m].get(metric)
            try:
                vals.append(float(v) if v is not None else np.nan)
            except Exception:
                vals.append(np.nan)
        data[metric] = np.array(vals, dtype=float)

    # Ordenar modelos por métrica principal (f1_macro) descendente para mejor legibilidad
    primary = 'f1_macro' if 'f1_macro' in metrics_to_show else metrics_to_show[0]
    order = np.argsort(-np.nan_to_num(data.get(primary, np.zeros(n_models)), nan=-1.0))
    model_names = [model_names[i] for i in order]
    for metric in metrics_to_show:
        data[metric] = data[metric][order]

    # Estética y paleta
    plt.style.use('ggplot')
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in range(len(metrics_to_show))]

    # Gráfico de barras horizontales agrupadas
    fig, ax = plt.subplots(figsize=(max(10, n_models * 0.9), 6))
    y = np.arange(n_models)
    height = 0.18

    for i, metric in enumerate(metrics_to_show):
        offsets = y + (i - (len(metrics_to_show)-1)/2) * height
        vals = data[metric]
        bars = ax.barh(offsets, vals, height, label=metric.replace('_', ' ').title(), color=colors[i])

        # Anotar dentro/fuera barra
        for bar, val in zip(bars, vals):
            if np.isnan(val):
                # marcar como ausente
                bx = bar.get_width()
                ax.text(bx + 0.02, bar.get_y() + bar.get_height()/2, 'N/A', va='center', fontsize=9, color='gray')
            else:
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f"{val:.3f}", va='center', fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(model_names)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel('Valor', fontsize=12, fontweight='bold')
    ax.set_title('Comparativa de Modelos — Accuracy / F1 / ROC-AUC', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    plt.tight_layout()

    out_path = os.path.join(plots_dir, 'models_comparison_improved.png')
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"  ✓ Imagen comparativa guardada: {out_path}")
