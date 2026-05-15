"""
Módulo de Regresión Logística Multiclase con estrategia One-vs-All (OvA).
Implementada completamente con numpy desde cero.
"""

import numpy as np
from utils.helpers import set_seeds
from sklearn.utils.class_weight import compute_class_weight


class LogisticRegressionOvA:
    """
    Regresión Logística Multiclase con estrategia One-vs-All (OvA).
    
    Cada clasificador binario utiliza:
    - Función de coste: cross-entropy binaria con regularización L2
    - Optimización: gradient descent con learning rate configurable
    - Características:
        * Implementación pura con numpy
        * Parámetros ajustables: learning_rate, lambda (regularización), n_iterations
        * Métodos: fit, predict_proba, predict
    
    Atributos:
    -----------
    learning_rate : float
        Tasa de aprendizaje para gradient descent
    lambda_reg : float
        Parámetro de regularización L2
    n_iterations : int
        Número de iteraciones de descenso de gradiente
    n_classes : int
        Número de clases
    weights : list de np.ndarray
        Pesos aprendidos por cada clasificador OvA
    intercepts : list de float
        Términos independientes (sesgos) por clasificador
    """
    
    def __init__(self, learning_rate=0.01, lambda_reg=0.01, n_iterations=1000):
        """
        Inicializa el modelo de regresión logística OvA.
        
        Parámetros:
        -----------
        learning_rate : float, default=0.01
            Tasa de aprendizaje para gradient descent
        lambda_reg : float, default=0.01
            Parámetro de regularización L2
        n_iterations : int, default=1000
            Número máximo de iteraciones
        """
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.n_iterations = n_iterations
        self.n_classes = None
        self.weights = None
        self.intercepts = None
        self.n_features = None
    
    def sigmoid(self, z):
        """
        Función sigmoide: σ(z) = 1 / (1 + exp(-z))
        
        Parámetros:
        -----------
        z : np.ndarray
            Valores de entrada
        
        Retorno:
        --------
        np.ndarray : Valores de la función sigmoide
        """
        # Clip para evitar overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def binary_cross_entropy(self, y_true, y_pred, weights):
        """
        Calcula la entropía cruzada binaria con regularización L2.
        
        Parámetros:
        -----------
        y_true : np.ndarray
            Etiquetas verdaderas (0 o 1)
        y_pred : np.ndarray
            Predicciones probabilísticas (entre 0 y 1)
        weights : np.ndarray
            Pesos del modelo
        
        Retorno:
        --------
        float : Valor del coste con regularización
        """
        n_samples = len(y_true)
        
        # Clip para evitar log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        
        # Entropía cruzada binaria
        ce = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        
        # Regularización L2
        l2_reg = (self.lambda_reg / (2 * n_samples)) * np.sum(weights ** 2)
        
        return ce + l2_reg
    
    def fit_binary_classifier(self, X, y_binary, class_weight=None):
        """
        Entrena un clasificador binario (vs. resto) usando gradient descent.
        
        Parámetros:
        -----------
        X : np.ndarray
            Matriz de features (n_samples, n_features)
        y_binary : np.ndarray
            Etiquetas binarias (0 o 1)
        class_weight : dict, optional
            Pesos para las clases {0: peso_0, 1: peso_1}
        
        Retorno:
        --------
        tuple : (weights, intercept)
        """
        n_samples, n_features = X.shape
        
        # Inicializar pesos y sesgo con pequeños valores aleatorios
        weights = np.random.randn(n_features) * 0.01
        intercept = 0.0
        
        # Gradient descent
        for iteration in range(self.n_iterations):
            # Forward pass: z = X * w + b
            z = np.dot(X, weights) + intercept
            
            # Predicciones: y_pred = sigmoid(z)
            y_pred = self.sigmoid(z)
            
            # Gradientes con pesos de clase
            error = y_pred - y_binary
            
            # Aplicar pesos de clase si existen
            if class_weight is not None:
                sample_weights = np.array([class_weight[int(y)] for y in y_binary])
                error = error * sample_weights
            
            dw = (1 / n_samples) * np.dot(X.T, error) + (self.lambda_reg / n_samples) * weights
            db = (1 / n_samples) * np.sum(error)
            
            # Actualización de pesos
            weights -= self.learning_rate * dw
            intercept -= self.learning_rate * db
        
        return weights, intercept
    
    def fit(self, X, y):
        """
        Entrena el modelo OvA (un clasificador binario por clase).
        
        Parámetros:
        -----------
        X : np.ndarray
            Matriz de features (n_samples, n_features)
        y : np.ndarray
            Etiquetas (0, 1, ..., n_classes-1)
        
        Retorno:
        --------
        self : Retorna la instancia para permitir encadenamiento
        """
        set_seeds(42)
        
        n_features = X.shape[1]
        self.n_features = n_features
        self.n_classes = len(np.unique(y))
        
        # Calcular pesos de clase para balancear
        unique_classes = np.unique(y)
        class_weights_sklearn = compute_class_weight('balanced', classes=unique_classes, y=y)
        class_weights_dict = {cls: weight for cls, weight in zip(unique_classes, class_weights_sklearn)}
        
        self.weights = []
        self.intercepts = []
        
        print(f"Entrenando Regresión Logística OvA ({self.n_classes} clases)...")
        print(f"  Pesos de clase: {class_weights_dict}")
        
        for class_idx in range(self.n_classes):
            # Crear etiquetas binarias: 1 si es la clase actual, 0 en otro caso
            y_binary = (y == class_idx).astype(int)
            
            # Crear pesos binarios para este clasificador OvA
            binary_weights = {
                1: class_weights_dict.get(class_idx, 1.0),
                0: 1.0
            }
            
            # Entrenar clasificador binario con pesos
            w, b = self.fit_binary_classifier(X, y_binary, class_weight=binary_weights)
            self.weights.append(w)
            self.intercepts.append(b)
            
            print(f"  ✓ Clasificador {class_idx} entrenado")
        
        return self
    
    def predict_proba(self, X):
        """
        Predice probabilidades para todas las clases.
        
        Parámetros:
        -----------
        X : np.ndarray
            Matriz de features (n_samples, n_features)
        
        Retorno:
        --------
        np.ndarray : Probabilidades por clase (n_samples, n_classes)
        """
        if self.weights is None:
            raise ValueError("Modelo no entrenado. Ejecuta fit() primero.")
        
        n_samples = X.shape[0]
        probas = np.zeros((n_samples, self.n_classes))
        
        # Calcular probabilidades para cada clase
        for class_idx in range(self.n_classes):
            z = np.dot(X, self.weights[class_idx]) + self.intercepts[class_idx]
            probas[:, class_idx] = self.sigmoid(z)
        
        # Normalizar para que sumen a 1 (softmax suave)
        probas = probas / (np.sum(probas, axis=1, keepdims=True) + 1e-15)
        
        return probas
    
    def predict(self, X):
        """
        Predice las clases para las muestras.
        
        Parámetros:
        -----------
        X : np.ndarray
            Matriz de features (n_samples, n_features)
        
        Retorno:
        --------
        np.ndarray : Clases predichas (0, 1, ..., n_classes-1)
        """
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)
    
    def score(self, X, y):
        """
        Calcula la precisión (accuracy) en el conjunto proporcionado.
        
        Parámetros:
        -----------
        X : np.ndarray
            Matriz de features
        y : np.ndarray
            Etiquetas verdaderas
        
        Retorno:
        --------
        float : Accuracy (0 a 1)
        """
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy
