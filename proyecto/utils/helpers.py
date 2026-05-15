"""
Módulo de funciones auxiliares para el proyecto de clasificación de ajedrez.
Contiene utilidades para:
  - Configuración de seeds (reproducibilidad)
  - Parsing de control de tiempo (increment_code)
  - Extracción de familias ECO
"""

import numpy as np
import pandas as pd
import random


def set_seeds(seed=42):
    """
    Establece seeds en numpy, random y python para reproducibilidad total.
    
    Parámetros:
    -----------
    seed : int, default=42
        Valor de la semilla
    """
    np.random.seed(seed)
    random.seed(seed)


def parse_increment_code(increment_code):
    """
    Parsea el control de tiempo increment_code (ej. "10+0", "5+3")
    para extraer base_time e increment como números.
    
    Parámetros:
    -----------
    increment_code : str
        Formato: "base+increment" (ej. "10+0")
    
    Retorno:
    --------
    tuple : (base_time, increment)
        Ambos como float/int
    
    Ejemplo:
    --------
    >>> parse_increment_code("10+0")
    (10, 0)
    >>> parse_increment_code("5+3")
    (5, 3)
    """
    try:
        parts = str(increment_code).split('+')
        if len(parts) == 2:
            base_time = float(parts[0].strip())
            increment = float(parts[1].strip())
            return base_time, increment
        else:
            # Si no tiene el formato esperado, retornar valores por defecto
            return np.nan, np.nan
    except (ValueError, AttributeError):
        return np.nan, np.nan


def get_eco_family(eco_code):
    """
    Extrae la familia ECO (primera letra) de un código ECO.
    Familias válidas: A, B, C, D, E
    
    Parámetros:
    -----------
    eco_code : str
        Código ECO (ej. "e4", "d40")
    
    Retorno:
    --------
    str : La primera letra del código ECO en mayúsculas, o 'Unknown'
    
    Ejemplo:
    --------
    >>> get_eco_family("e4")
    'E'
    >>> get_eco_family("d40")
    'D'
    """
    try:
        if pd.isna(eco_code):
            return 'Unknown'
        first_letter = str(eco_code)[0].upper()
        if first_letter in ['A', 'B', 'C', 'D', 'E']:
            return first_letter
        else:
            return 'Unknown'
    except (AttributeError, IndexError, TypeError):
        return 'Unknown'


# Función para validación de datos
def check_missing_values(df):
    """
    Analiza y reporta valores faltantes en un DataFrame.
    
    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame a analizar
    
    Retorno:
    --------
    pd.Series : Conteo de valores faltantes por columna
    """
    import pandas as pd
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("Valores faltantes encontrados:")
        print(missing[missing > 0])
    else:
        print("No hay valores faltantes.")
    return missing


def balance_check(y, class_names=None):
    """
    Verifica el balance de clases en el target.
    
    Parámetros:
    -----------
    y : array-like
        Vector de target
    class_names : list, optional
        Nombres de las clases para display mejorado
    
    Retorno:
    --------
    dict : Contado de instancias por clase
    """
    from collections import Counter
    
    counts = Counter(y)
    total = len(y)
    
    print("Distribución de clases:")
    for class_label, count in sorted(counts.items()):
        pct = 100 * count / total
        if class_names and class_label < len(class_names):
            class_name = class_names[class_label]
        else:
            class_name = f"Clase {class_label}"
        print(f"  {class_name}: {count} muestras ({pct:.2f}%)")
    
    return dict(counts)
