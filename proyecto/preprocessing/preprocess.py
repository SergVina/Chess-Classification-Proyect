"""
Módulo de preprocesamiento y análisis exploratorio de datos (EDA).
Contiene funciones para:
  - Carga del dataset
  - Muestreo estratificado
  - Construcción del target (skill_level discretizado)
  - Feature engineering
  - División train/val/test
  - Generación de visualizaciones EDA
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Backend non-interactive (sin tkinter warnings)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from utils.helpers import set_seeds, parse_increment_code, get_eco_family


def load_data(csv_path):
    """
    Carga el dataset de ajedrez desde un archivo CSV.
    
    Parámetros:
    -----------
    csv_path : str
        Ruta al archivo games.csv
    
    Retorno:
    --------
    pd.DataFrame : Dataset cargado
    """
    print(f"Cargando datos desde {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"  Forma inicial: {df.shape}")
    print(f"  Columnas: {list(df.columns)}")
    return df


def create_target(df):
    """
    Crea la variable target (skill_level) basada en skill_level = (white_rating + black_rating) / 2
    discretizado en 3 clases usando percentiles 33 y 66.
    
    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame con columnas white_rating y black_rating
    
    Retorno:
    --------
    np.ndarray : Vector target discretizado (0=Principiante, 1=Intermedio, 2=Avanzado)
    """
    # Calcular skill_level promedio
    skill_level_avg = (df['white_rating'] + df['black_rating']) / 2
    
    # Obtener percentiles 33 y 66 del conjunto COMPLETO (antes del muestreo)
    p33 = np.percentile(skill_level_avg, 33)
    p66 = np.percentile(skill_level_avg, 66)
    
    print(f"Percentil 33: {p33:.2f}, Percentil 66: {p66:.2f}")
    
    # Discretizar en 3 clases
    target = np.digitize(skill_level_avg, bins=[p33, p66], right=False)
    # digitize retorna 1 para valores < p33, 2 para p33 <= x < p66, 3 para >= p66
    # Necesitamos 0, 1, 2
    target = target - 1
    target = np.clip(target, 0, 2)
    
    return target, skill_level_avg, p33, p66


def stratified_sample(df, target, n_samples=2500):
    """
    Realiza un muestreo estratificado para mantener proporciones de clase.
    
    Parámetros:
    -----------
    df : pd.DataFrame
        Dataset completo
    target : np.ndarray
        Vector target
    n_samples : int, default=2500
        Número de muestras a extraer (entre 2500 y 3000)
    
    Retorno:
    --------
    tuple : (df_sampled, target_sampled, indices)
    """
    print(f"\nRealizando muestreo estratificado para {n_samples} partidas...")
    
    # Usar train_test_split para muestreo estratificado
    indices_keep = np.arange(len(df))
    indices_sampled, _, _, _ = train_test_split(
        indices_keep, 
        target,
        train_size=n_samples,
        stratify=target,
        random_state=42
    )
    
    indices_sampled = np.sort(indices_sampled)
    df_sampled = df.iloc[indices_sampled].reset_index(drop=True)
    target_sampled = target[indices_sampled]
    
    print(f"  Muestreo completado: {df_sampled.shape[0]} partidas")
    print(f"  Distribución de clases en muestra:")
    for cls in [0, 1, 2]:
        count = np.sum(target_sampled == cls)
        pct = 100 * count / len(target_sampled)
        print(f"    Clase {cls}: {count} ({pct:.1f}%)")
    
    return df_sampled, target_sampled, indices_sampled


def feature_engineering(df):
    """
    Realiza feature engineering extrayendo y transformando las 7 features principales.
    
    Parámetros:
    -----------
    df : pd.DataFrame
        Dataset muestreado
    
    Retorno:
    --------
    dict : Diccionario con features preprocesadas y metadata de encoders/scalers
    """
    features_data = {}
    
    # Feature 1: turns (numérico)
    features_data['turns'] = df['turns'].values.reshape(-1, 1)
    print("✓ Feature 'turns' extraída")
    
    # Feature 2: opening_ply (numérico)
    features_data['opening_ply'] = df['opening_ply'].values.reshape(-1, 1)
    print("✓ Feature 'opening_ply' extraída")
    
    # Feature 3: victory_status (categórico → One-Hot)
    victory_status_ohe = pd.get_dummies(df['victory_status'], prefix='victory')
    features_data['victory_status_cols'] = victory_status_ohe.columns.tolist()
    features_data['victory_status'] = victory_status_ohe.values
    print(f"✓ Feature 'victory_status' One-Hot Encoded: {len(features_data['victory_status_cols'])} categorías")
    
    # Feature 4: winner (categórico → One-Hot)
    winner_ohe = pd.get_dummies(df['winner'], prefix='winner')
    features_data['winner_cols'] = winner_ohe.columns.tolist()
    features_data['winner'] = winner_ohe.values
    print(f"✓ Feature 'winner' One-Hot Encoded: {len(features_data['winner_cols'])} categorías")
    
    # Feature 5: rated (binario 0/1)
    features_data['rated'] = (df['rated'].astype(int)).values.reshape(-1, 1)
    print("✓ Feature 'rated' extraída (binaria)")
    
    # Feature 6: increment_code (extraer base_time e increment)
    base_times = []
    increments = []
    for ic in df['increment_code']:
        base, incr = parse_increment_code(ic)
        base_times.append(base)
        increments.append(incr)
    features_data['base_time'] = np.array(base_times).reshape(-1, 1)
    features_data['increment'] = np.array(increments).reshape(-1, 1)
    print("✓ Feature 'base_time' e 'increment' extraídas de increment_code")
    
    # Feature 7: opening_eco (agrupar por familia ECO: A, B, C, D, E)
    eco_families = []
    for eco in df['opening_eco']:
        family = get_eco_family(eco)
        eco_families.append(family)
    eco_df = pd.DataFrame({'eco_family': eco_families})
    eco_ohe = pd.get_dummies(eco_df['eco_family'], prefix='eco')
    features_data['eco_families_cols'] = eco_ohe.columns.tolist()
    features_data['eco_families'] = eco_ohe.values
    print(f"✓ Feature 'opening_eco' agrupada por familia: {len(features_data['eco_families_cols'])} familias")
    
    return features_data


def create_feature_matrix(features_data):
    """
    Concatena todas las features en una matriz X.
    
    Parámetros:
    -----------
    features_data : dict
        Diccionario con features preprocesadas
    
    Retorno:
    --------
    np.ndarray : Matriz de features (n_samples, n_features)
    """
    feature_list = [
        features_data['turns'],
        features_data['opening_ply'],
        features_data['victory_status'],
        features_data['winner'],
        features_data['rated'],
        features_data['base_time'],
        features_data['increment'],
        features_data['eco_families']
    ]
    
    X = np.hstack(feature_list)
    print(f"\nMatriz de features creada: {X.shape}")
    return X


def preprocess_data(df, target, features_data, fit_scalers=True):
    """
    Preprocesa las features: normaliza numéricas con StandardScaler.
    
    Parámetros:
    -----------
    df : pd.DataFrame
        Dataset
    target : np.ndarray
        Target
    features_data : dict
        Features extraídas
    fit_scalers : bool
        Si True, ajusta los scalers. Si False, solo transforma.
    
    Retorno:
    --------
    tuple : (X_processed, scalers_dict)
    """
    scalers = {}
    
    # Indices de columnas numéricas en la matriz final
    # turns (0), opening_ply (1), base_time, increment, resto es one-hot
    numeric_features = ['turns', 'opening_ply', 'base_time', 'increment', 'rated']
    numeric_indices = [0, 1, 5, 6, 4]  # Posiciones en X después del hstack
    
    # NOTE: This legacy function is kept for compatibility but the preferred
    # workflow is to call `fit_and_apply_scalers` after splitting so scalers
    # are fit only on the training set (avoid data leakage).
    X = create_feature_matrix(features_data)
    
    # Para normalización, necesitamos procesar los features numéricos
    X_processed = X.copy()
    
    # Normalizar turns
    scaler_turns = StandardScaler()
    if fit_scalers:
        X_processed[:, :1] = scaler_turns.fit_transform(X[:, :1])
    else:
        X_processed[:, :1] = scaler_turns.transform(X[:, :1])
    scalers['turns'] = scaler_turns
    
    # Normalizar opening_ply
    scaler_opening_ply = StandardScaler()
    if fit_scalers:
        X_processed[:, 1:2] = scaler_opening_ply.fit_transform(X[:, 1:2])
    else:
        X_processed[:, 1:2] = scaler_opening_ply.transform(X[:, 1:2])
    scalers['opening_ply'] = scaler_opening_ply
    
    # victory_status one-hot: 2:6 aprox (depende de cuántas están, generalmente 4)
    n_victory = len(features_data['victory_status_cols'])
    n_winner = len(features_data['winner_cols'])
    
    # rated está en: 2 + n_victory + n_winner
    idx_rated = 2 + n_victory + n_winner
    
    # Normalizar rated
    scaler_rated = StandardScaler()
    if fit_scalers:
        X_processed[:, idx_rated:idx_rated+1] = scaler_rated.fit_transform(X[:, idx_rated:idx_rated+1])
    else:
        X_processed[:, idx_rated:idx_rated+1] = scaler_rated.transform(X[:, idx_rated:idx_rated+1])
    scalers['rated'] = scaler_rated
    
    # base_time: después de rated
    idx_base_time = idx_rated + 1
    scaler_base_time = StandardScaler()
    if fit_scalers:
        X_processed[:, idx_base_time:idx_base_time+1] = scaler_base_time.fit_transform(X[:, idx_base_time:idx_base_time+1])
    else:
        X_processed[:, idx_base_time:idx_base_time+1] = scaler_base_time.transform(X[:, idx_base_time:idx_base_time+1])
    scalers['base_time'] = scaler_base_time
    
    # increment: después de base_time
    idx_increment = idx_base_time + 1
    scaler_increment = StandardScaler()
    if fit_scalers:
        X_processed[:, idx_increment:idx_increment+1] = scaler_increment.fit_transform(X[:, idx_increment:idx_increment+1])
    else:
        X_processed[:, idx_increment:idx_increment+1] = scaler_increment.transform(X[:, idx_increment:idx_increment+1])
    scalers['increment'] = scaler_increment
    
    return X_processed, scalers


def fit_and_apply_scalers(X_full, X_train_idx, features_data):
    """
    Ajusta los StandardScalers usando únicamente `X_full[X_train_idx]` y
    transforma `X_full` entero. Retorna (X_processed_full, scalers_dict).

    Parámetros:
    - X_full: ndarray (n_samples, n_features) sin normalizar
    - X_train_idx: array-like índices de las muestras de entrenamiento
    - features_data: dict devuelto por feature_engineering (necesario para
      calcular posiciones de columnas)
    """
    scalers = {}
    X_processed = X_full.copy()

    n_victory = len(features_data['victory_status_cols'])
    n_winner = len(features_data['winner_cols'])

    idx_rated = 2 + n_victory + n_winner
    idx_base_time = idx_rated + 1
    idx_increment = idx_base_time + 1

    # Columns numéricas a normalizar: turns (0), opening_ply (1), rated,
    # base_time, increment
    # Fit scalers on TRAIN only
    scaler_turns = StandardScaler()
    scaler_turns.fit(X_full[X_train_idx, 0:1])
    X_processed[:, 0:1] = scaler_turns.transform(X_full[:, 0:1])
    scalers['turns'] = scaler_turns

    scaler_opening_ply = StandardScaler()
    scaler_opening_ply.fit(X_full[X_train_idx, 1:2])
    X_processed[:, 1:2] = scaler_opening_ply.transform(X_full[:, 1:2])
    scalers['opening_ply'] = scaler_opening_ply

    scaler_rated = StandardScaler()
    scaler_rated.fit(X_full[X_train_idx, idx_rated:idx_rated+1])
    X_processed[:, idx_rated:idx_rated+1] = scaler_rated.transform(X_full[:, idx_rated:idx_rated+1])
    scalers['rated'] = scaler_rated

    scaler_base_time = StandardScaler()
    scaler_base_time.fit(X_full[X_train_idx, idx_base_time:idx_base_time+1])
    X_processed[:, idx_base_time:idx_base_time+1] = scaler_base_time.transform(X_full[:, idx_base_time:idx_base_time+1])
    scalers['base_time'] = scaler_base_time

    scaler_increment = StandardScaler()
    scaler_increment.fit(X_full[X_train_idx, idx_increment:idx_increment+1])
    X_processed[:, idx_increment:idx_increment+1] = scaler_increment.transform(X_full[:, idx_increment:idx_increment+1])
    scalers['increment'] = scaler_increment

    return X_processed, scalers


def split_data(X, y, test_size=0.2, val_size=0.2):
    """
    Divide los datos en train (60%), validación (20%) y test (20%).
    Mantiene proporciones de clase con stratify.
    
    Parámetros:
    -----------
    X : np.ndarray
        Matriz de features
    y : np.ndarray
        Target
    test_size : float
        Proporción para test
    val_size : float
        Proporción para validación (sobre train+val)
    
    Retorno:
    --------
    dict : Diccionario con X_train, X_val, X_test, y_train, y_val, y_test
    """
    # Hacemos split sobre índices para poder devolver también los índices
    indices = np.arange(len(X))

    # Primera división: temp vs test (80% vs 20%)
    idx_temp, idx_test = train_test_split(
        indices, test_size=test_size, stratify=y, random_state=42
    )

    # Segunda división: train vs val (sobre temp)
    val_proportion = val_size / (1 - test_size)
    idx_train, idx_val = train_test_split(
        idx_temp, test_size=val_proportion, stratify=y[idx_temp], random_state=42
    )

    X_train = X[idx_train]
    X_val = X[idx_val]
    X_test = X[idx_test]

    y_train = y[idx_train]
    y_val = y[idx_val]
    y_test = y[idx_test]
    
    print(f"\nDivisión de datos:")
    print(f"  Train: {X_train.shape[0]} muestras ({100*X_train.shape[0]/len(X):.1f}%)")
    print(f"  Validación: {X_val.shape[0]} muestras ({100*X_val.shape[0]/len(X):.1f}%)")
    print(f"  Test: {X_test.shape[0]} muestras ({100*X_test.shape[0]/len(X):.1f}%)")
    
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'idx_train': idx_train,
        'idx_val': idx_val,
        'idx_test': idx_test
    }


def plot_eda(df, y, skill_level_avg, p33, p66, plots_dir='plots'):
    """
    Genera 6 gráficos EDA y los guarda en plots_dir.
    
    Parámetros:
    -----------
    df : pd.DataFrame
        Dataset muestreado
    y : np.ndarray
        Target discretizado
    skill_level_avg : np.ndarray
        Skill level promedio continuo
    p33, p66 : float
        Percentiles para visualización
    plots_dir : str
        Directorio para guardar plots
    """
    os.makedirs(plots_dir, exist_ok=True)
    
    class_names = ['Principiante', 'Intermedio', 'Avanzado']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    # ===== GRÁFICO 1: Histograma de ELO promedio con percentiles =====
    print("\nGenerando gráficos EDA...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for cls in [0, 1, 2]:
        mask = y == cls
        ax.hist(skill_level_avg[mask], bins=30, alpha=0.6, label=class_names[cls], color=colors[cls])
    
    ax.axvline(p33, color='red', linestyle='--', linewidth=2, label=f'Percentil 33 ({p33:.0f})')
    ax.axvline(p66, color='green', linestyle='--', linewidth=2, label=f'Percentil 66 ({p66:.0f})')
    
    ax.set_xlabel('ELO Promedio (white_rating + black_rating) / 2', fontsize=12)
    ax.set_ylabel('Frecuencia', fontsize=12)
    ax.set_title('Distribución de ELO Promedio por Nivel de Habilidad', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '01_elo_distribution.png'), dpi=150)
    plt.close()
    print("  ✓ 01_elo_distribution.png")
    
    # Conclusión: Los percentiles dividen correctamente el dataset en 3 clases
    
    # ===== GRÁFICO 2: Distribución de clases =====
    fig, ax = plt.subplots(figsize=(8, 6))
    
    counts = [np.sum(y == cls) for cls in [0, 1, 2]]
    bars = ax.bar(class_names, counts, color=colors, edgecolor='black', linewidth=1.5)
    
    # Añadir valores en las barras
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({100*count/len(y):.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Número de Partidas', fontsize=12)
    ax.set_title('Distribución de Clases (Balanceo)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '02_class_distribution.png'), dpi=150)
    plt.close()
    print("  ✓ 02_class_distribution.png")
    
    # Conclusión: El muestreo estratificado mantiene proporciones de clase aproximadamente iguales
    
    # ===== GRÁFICO 3: Boxplots de turns y opening_ply por clase =====
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    turns_data = [df[y == cls]['turns'].values for cls in [0, 1, 2]]
    opening_ply_data = [df[y == cls]['opening_ply'].values for cls in [0, 1, 2]]
    
    bp1 = axes[0].boxplot(turns_data, labels=class_names, patch_artist=True)
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
    axes[0].set_ylabel('Número de Movimientos', fontsize=11)
    axes[0].set_title('Distribución de Movimientos por Nivel', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    bp2 = axes[1].boxplot(opening_ply_data, labels=class_names, patch_artist=True)
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
    axes[1].set_ylabel('Movimientos de Apertura', fontsize=11)
    axes[1].set_title('Distribución de Movimientos de Apertura por Nivel', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '03_turns_opening_ply_boxplot.png'), dpi=150)
    plt.close()
    print("  ✓ 03_turns_opening_ply_boxplot.png")
    
    # Conclusión: Jugadores avanzados tienden a jugar partidas más largas
    
    # ===== GRÁFICO 4: victory_status agrupado por clase =====
    fig, ax = plt.subplots(figsize=(10, 6))
    
    victory_status_dist = {}
    for cls in [0, 1, 2]:
        mask = y == cls
        dist = df[mask]['victory_status'].value_counts()
        victory_status_dist[class_names[cls]] = dist
    
    victory_status_df = pd.DataFrame(victory_status_dist).fillna(0)
    victory_status_df.plot(kind='bar', ax=ax, color=colors, edgecolor='black', linewidth=1)
    
    ax.set_xlabel('Estado de Finalización', fontsize=12)
    ax.set_ylabel('Cantidad de Partidas', fontsize=12)
    ax.set_title('Distribución de victory_status por Nivel de Habilidad', fontsize=14, fontweight='bold')
    ax.legend(title='Nivel', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '04_victory_status_by_class.png'), dpi=150)
    plt.close()
    print("  ✓ 04_victory_status_by_class.png")
    
    # Conclusión: La modalidad de finalización varía según el nivel
    
    # ===== GRÁFICO 5: Matriz de correlación =====
    # Incluir features numéricas: turns, opening_ply, y luego los binarios/discretos
    numeric_cols = ['turns', 'opening_ply', 'white_rating', 'black_rating']
    numeric_data = df[numeric_cols].values
    
    corr_matrix = np.corrcoef(numeric_data.T)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    
    ax.set_xticks(range(len(numeric_cols)))
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=45, ha='right')
    ax.set_yticklabels(numeric_cols)
    
    # Añadir valores de correlación en las células
    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                   ha='center', va='center',
                   color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black',
                   fontsize=10, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlación de Pearson', fontsize=11)
    ax.set_title('Matriz de Correlación de Features Numéricas', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '05_correlation_matrix.png'), dpi=150)
    plt.close()
    print("  ✓ 05_correlation_matrix.png")
    
    # Conclusión: white_rating y black_rating están fuertemente correlacionadas
    
    # ===== GRÁFICO 6: Distribución de base_time por clase =====
    fig, ax = plt.subplots(figsize=(10, 6))
    
    base_times = []
    classes = []
    for cls in [0, 1, 2]:
        mask = y == cls
        times = []
        for ic in df[mask]['increment_code']:
            base_time, _ = parse_increment_code(ic)
            if not np.isnan(base_time):
                times.append(base_time)
        base_times.append(times)
        classes.extend([class_names[cls]] * len(times))
    
    # Flatten para plot
    all_times = []
    all_labels = []
    for i, times in enumerate(base_times):
        all_times.extend(times)
        all_labels.extend([class_names[i]] * len(times))
    
    # Crear boxplot
    data_by_class = [base_times[0], base_times[1], base_times[2]]
    bp = ax.boxplot(data_by_class, labels=class_names, patch_artist=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Tiempo Base (segundos)', fontsize=12)
    ax.set_title('Distribución del Control de Tiempo (base) por Nivel de Habilidad', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '06_base_time_by_class.png'), dpi=150)
    plt.close()
    print("  ✓ 06_base_time_by_class.png")
    print("\nGráficos EDA completados.")
    
    # Conclusión: Diferentes controles de tiempo se prefieren en diferentes niveles


def preprocess_pipeline(csv_path, n_samples=2500, plots_dir='plots', skip_eda=False):
    """
    Pipeline completo de preprocesamiento.
    
    Parámetros:
    -----------
    csv_path : str
        Ruta al archivo games.csv
    n_samples : int
        Número de partidas a muestrear
    plots_dir : str
        Directorio para guardar gráficos
    skip_eda : bool
        Si True, no genera gráficos EDA
    
    Retorno:
    --------
    dict : Diccionario completo con datos preprocesados y metadata
    """
    set_seeds(42)
    
    # 1. Cargar datos
    df = load_data(csv_path)
    
    # 2. Crear target
    target, skill_level_avg, p33, p66 = create_target(df)
    
    # 3. Muestreo estratificado
    df_sampled, target_sampled, indices_sampled = stratified_sample(df, target, n_samples=n_samples)
    
    # Actualizar skill_level_avg para el dataset muestreado
    skill_level_avg_sampled = skill_level_avg.iloc[indices_sampled].values if isinstance(skill_level_avg, pd.Series) else skill_level_avg[indices_sampled]
    
    # 4. Feature engineering
    features_data = feature_engineering(df_sampled)
    
    # 5. Crear matriz de features (sin normalizar) y hacer SPLIT primero
    X_raw = create_feature_matrix(features_data)

    # 6. Split train/val/test (antes de ajustar scalers para evitar data leakage)
    data_split = split_data(X_raw, target_sampled, test_size=0.2, val_size=0.2)

    # 7. Ajustar scalers usando SOLO el TRAIN y transformar todo el conjunto
    idx_train = data_split['idx_train']
    X_processed, scalers = fit_and_apply_scalers(X_raw, idx_train, features_data)

    # Reemplazar X_train/X_val/X_test en data_split con sus versiones normalizadas
    data_split['X_train'] = X_processed[data_split['idx_train']]
    data_split['X_val'] = X_processed[data_split['idx_val']]
    data_split['X_test'] = X_processed[data_split['idx_test']]
    
    # 7. EDA (si no está skipped)
    if not skip_eda:
        plot_eda(df_sampled, target_sampled, skill_level_avg_sampled, p33, p66, plots_dir=plots_dir)
    
    # Retornar diccionario con toda la información
    result = {
        'df_original': df,
        'df_sampled': df_sampled,
        'target': target_sampled,
        'X': X_processed,
        'scalers': scalers,
        'features_data': features_data,
        'data_split': data_split,
        'p33': p33,
        'p66': p66,
        'skill_level_avg': skill_level_avg_sampled
    }
    
    return result
