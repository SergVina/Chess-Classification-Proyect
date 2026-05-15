"""
Módulo de Red Neuronal Feedforward completamente implementada con numpy.
Características principales:
  - Arquitectura configurable (capas ocultas flexibles)
  - Activación ReLU en capas ocultas
  - Activacion Softmax en capa de salida
  - Backpropagation completo
  - Early stopping
  - Regularización L2
  - Minibatch SGD
"""

import numpy as np
from utils.helpers import set_seeds
from sklearn.utils.class_weight import compute_class_weight


class NeuralNetwork:
    """
    Red neuronal feedforward multiclase con arquitectura configurable.
    
    Arquitectura:
    - Capa de entrada: n_features neuronas
    - Capas ocultas: configurables (ej. [64, 32])
    - Capa de salida: 3 neuronas (para 3 clases) con softmax
    
    Atributos:
    -----------
    hidden_layers : list de int
        Lista con el número de neuronas en cada capa oculta
    learning_rate : float
        Tasa de aprendizaje
    lambda_reg : float
        Parámetro de regularización L2
    batch_size : int
        Tamaño de minibatch
    max_epochs : int
        Número máximo de épocas
    patience : int
        Paciencia para early stopping
    weights : list de np.ndarray
        Pesos de todas las capas
    biases : list de np.ndarray
        Sesgos de todas las capas
    history : dict
        Historial de loss y accuracy por época
    """
    
    def __init__(self, hidden_layers=[64, 32], learning_rate=0.01, lambda_reg=0.01,
                 batch_size=32, max_epochs=200, patience=20, random_state=42):
        """
        Inicializa la red neuronal.
        
        Parámetros:
        -----------
        hidden_layers : list de int, default=[64, 32]
            Número de neuronas en cada capa oculta
        learning_rate : float, default=0.01
            Tasa de aprendizaje
        lambda_reg : float, default=0.01
            Regularización L2
        batch_size : int, default=32
            Tamaño del minibatch
        max_epochs : int, default=200
            Máximo de épocas
        patience : int, default=20
            Paciencia para early stopping
        random_state : int, default=42
            Seed para reproducibilidad
        """
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.random_state = random_state
        
        self.weights = None
        self.biases = None
        self.class_weights = None
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        self.best_epoch = None
        self.best_val_loss = np.inf
    
    def he_initialization(self, n_in, n_out):
        """
        Inicialización He para capas ReLU (recomendado para ReLU).
        
        W ~ N(0, sqrt(2/n_in))
        
        Parámetros:
        -----------
        n_in : int
            Número de neuronas de entrada
        n_out : int
            Número de neuronas de salida
        
        Retorno:
        --------
        np.ndarray : Matriz de pesos inicializados
        """
        return np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
    
    def initialize_weights(self, n_input_features, n_classes):
        """
        Inicializa todos los pesos y sesgos de la red.
        
        Parámetros:
        -----------
        n_input_features : int
            Número de features de entrada
        n_classes : int
            Número de clases de salida (3 para este proyecto)
        """
        set_seeds(self.random_state)
        
        # Arquitectura: input -> hidden_layers -> output
        layer_dims = [n_input_features] + self.hidden_layers + [n_classes]
        
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_dims) - 1):
            n_in = layer_dims[i]
            n_out = layer_dims[i + 1]
            
            # Inicializar pesos con He para todas las capas
            w = self.he_initialization(n_in, n_out)
            b = np.zeros((1, n_out))
            
            self.weights.append(w)
            self.biases.append(b)
    
    def relu(self, z):
        """
        Activación ReLU: max(0, z)
        
        Parámetro:
        -----------
        z : np.ndarray
            Entrada
        
        Retorno:
        --------
        np.ndarray : Salida ReLU
        """
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        """
        Derivada de ReLU: 1 si z > 0, else 0
        
        Parámetro:
        -----------
        z : np.ndarray
            Entrada (valores pre-activación)
        
        Retorno:
        --------
        np.ndarray : Derivada
        """
        return (z > 0).astype(float)
    
    def softmax(self, z):
        """
        Función Softmax para multiclase: exp(z_i) / sum(exp(z))
        
        Parámetro:
        -----------
        z : np.ndarray
            Entrada (n_samples, n_classes)
        
        Retorno:
        --------
        np.ndarray : Probabilidades normalizadas
        """
        # Restar max para estabilidad numérica
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def forward_pass(self, X):
        """
        Passes los datos a través de la red (forward propagation).
        
        Parámetros:
        -----------
        X : np.ndarray
            Matriz de features (n_samples, n_features)
        
        Retorno:
        --------
        tuple : (a_list, z_list)
            - a_list: activaciones de cada capa
            - z_list: valores pre-activación de cada capa
        """
        n_layers = len(self.weights)
        a = X  # Entrada es la activación de la capa 0
        a_list = [a]
        z_list = []
        
        # Capas ocultas con ReLU
        for layer in range(n_layers - 1):
            z = np.dot(a, self.weights[layer]) + self.biases[layer]
            a = self.relu(z)
            z_list.append(z)
            a_list.append(a)
        
        # Capa de salida con Softmax
        z_output = np.dot(a, self.weights[-1]) + self.biases[-1]
        a_output = self.softmax(z_output)
        z_list.append(z_output)
        a_list.append(a_output)
        
        return a_list, z_list
    
    def backward_pass(self, X, y, a_list, z_list):
        """
        Realiza backpropagation para calcular gradientes.
        
        Parámetros:
        -----------
        X : np.ndarray
            Features originales
        y : np.ndarray
            Etiquetas (0, 1, 2 para multiclase)
        a_list : list
            Activaciones de cada capa
        z_list : list
            Valores pre-activación de cada capa
        
        Retorno:
        --------
        tuple : (dW_list, dB_list)
            - dW_list: gradientes de pesos
            - dB_list: gradientes de sesgos
        """
        n_samples = X.shape[0]
        n_layers = len(self.weights)
        
        # Convertir y a one-hot encoding
        y_onehot = np.eye(a_list[-1].shape[1])[y]
        
        # Inicializar listas de gradientes
        dW_list = [None] * n_layers
        dB_list = [None] * n_layers
        
        # ===== Capa de salida: Softmax + Cross-entropy =====
        # Derivada de (softmax + cross-entropy) = predictions - y_true
        delta = a_list[-1] - y_onehot
        
        # Gradiente de pesos capa de salida
        dW = np.dot(a_list[-2].T, delta) / n_samples + (self.lambda_reg / n_samples) * self.weights[-1]
        dB = np.sum(delta, axis=0, keepdims=True) / n_samples
        
        dW_list[-1] = dW
        dB_list[-1] = dB
        
        # ===== Capas ocultas: Backprop con ReLU =====
        # Comenzar desde la penúltima capa (índice n_layers - 2)
        for layer in range(n_layers - 2, -1, -1):
            # Derivada de ReLU
            relu_deriv = self.relu_derivative(z_list[layer])
            
            # Propagar delta a través de la derivada
            delta = np.dot(delta, self.weights[layer + 1].T) * relu_deriv
            
            # Calcular gradientes
            dW = np.dot(a_list[layer].T, delta) / n_samples + (self.lambda_reg / n_samples) * self.weights[layer]
            dB = np.sum(delta, axis=0, keepdims=True) / n_samples
            
            dW_list[layer] = dW
            dB_list[layer] = dB
        
        return dW_list, dB_list
    
    def compute_loss(self, X, y):
        """
        Calcula la función de coste (cross-entropy + regularización L2).
        
        Parámetros:
        -----------
        X : np.ndarray
            Features
        y : np.ndarray
            Etiquetas
        
        Retorno:
        --------
        float : Valor del coste
        """
        a_list, _ = self.forward_pass(X)
        
        # One-hot encoding de y
        y_onehot = np.eye(a_list[-1].shape[1])[y]
        
        # Cross-entropy con pesos de clase
        n_samples = X.shape[0]
        predictions = a_list[-1]
        predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
        
        # Aplicar pesos de clase si existen
        if self.class_weights is not None:
            sample_weights = np.array([self.class_weights[yi] for yi in y])
            ce_loss = -np.sum(y_onehot * np.log(predictions) * sample_weights.reshape(-1, 1)) / n_samples
        else:
            ce_loss = -np.sum(y_onehot * np.log(predictions)) / n_samples
        
        # Regularización L2
        l2_reg = 0
        for w in self.weights:
            l2_reg += np.sum(w ** 2)
        l2_reg = (self.lambda_reg / (2 * n_samples)) * l2_reg
        
        return ce_loss + l2_reg
    
    def compute_accuracy(self, X, y):
        """
        Calcula la precisión (accuracy).
        
        Parámetros:
        -----------
        X : np.ndarray
            Features
        y : np.ndarray
            Etiquetas
        
        Retorno:
        --------
        float : Accuracy (0 a 1)
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def fit(self, X_train, y_train, X_val, y_val):
        """
        Entrena la red neuronal con early stopping basado en validación.
        
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
        self : Retorna la instancia para encadenamiento
        """
        set_seeds(self.random_state)
        
        # Calcular pesos de clase para balancear
        unique_classes = np.unique(y_train)
        class_weights_sklearn = compute_class_weight('balanced', classes=unique_classes, y=y_train)
        self.class_weights = {cls: weight for cls, weight in zip(unique_classes, class_weights_sklearn)}
        
        # Inicializar pesos
        self.initialize_weights(X_train.shape[1], len(np.unique(y_train)))
        
        n_samples = X_train.shape[0]
        n_batches = max(1, n_samples // self.batch_size)
        
        patience_counter = 0
        self.best_epoch = 0
        self.best_val_loss = np.inf
        
        print(f"Entrenando Red Neuronal con arquitectura {self.hidden_layers}...")
        print(f"  Pesos de clase: {self.class_weights}")
        
        for epoch in range(self.max_epochs):
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]
            
            train_loss = 0
            train_acc = 0
            
            # Minibatches
            for batch_idx in range(n_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, n_samples)
                
                X_batch = X_train_shuffled[start_idx:end_idx]
                y_batch = y_train_shuffled[start_idx:end_idx]
                
                # Forward pass
                a_list, z_list = self.forward_pass(X_batch)
                
                # Backward pass
                dW_list, dB_list = self.backward_pass(X_batch, y_batch, a_list, z_list)
                
                # Actualizar pesos
                for layer in range(len(self.weights)):
                    self.weights[layer] -= self.learning_rate * dW_list[layer]
                    self.biases[layer] -= self.learning_rate * dB_list[layer]
            
            # Evaluar en train completo
            train_loss = self.compute_loss(X_train, y_train)
            train_acc = self.compute_accuracy(X_train, y_train)
            
            # Evaluar en validación
            val_loss = self.compute_loss(X_val, y_val)
            val_acc = self.compute_accuracy(X_val, y_val)
            
            # Guardar historial
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                patience_counter = 0
                
                # Guardar mejor modelo
                self.best_weights = [w.copy() for w in self.weights]
                self.best_biases = [b.copy() for b in self.biases]
            else:
                patience_counter += 1
            
            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f"  Época {epoch+1}: Loss_train={train_loss:.4f}, Loss_val={val_loss:.4f}, "
                      f"Acc_train={train_acc:.4f}, Acc_val={val_acc:.4f}")
            
            # Early stopping
            if patience_counter >= self.patience:
                print(f"Early stopping en época {epoch+1} (paciencia alcanzada)")
                # Restaurar mejores pesos
                self.weights = self.best_weights
                self.biases = self.best_biases
                break
        
        print(f"Entrenamiento completado. Mejor época: {self.best_epoch+1}")
        return self
    
    def predict_proba(self, X):
        """
        Predice probabilidades para todas las clases.
        
        Parámetros:
        -----------
        X : np.ndarray
            Features
        
        Retorno:
        --------
        np.ndarray : Probabilidades (n_samples, n_classes)
        """
        a_list, _ = self.forward_pass(X)
        return a_list[-1]
    
    def predict(self, X):
        """
        Predice las clases.
        
        Parámetros:
        -----------
        X : np.ndarray
            Features
        
        Retorno:
        --------
        np.ndarray : Clases predichas
        """
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)
    
    def score(self, X, y):
        """
        Calcula accuracy.
        
        Parámetros:
        -----------
        X : np.ndarray
            Features
        y : np.ndarray
            Etiquetas
        
        Retorno:
        --------
        float : Accuracy
        """
        return self.compute_accuracy(X, y)
