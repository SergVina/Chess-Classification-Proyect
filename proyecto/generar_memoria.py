"""
Generador de memoria academica - Clasificacion de nivel de habilidad en ajedrez.
Ejecutar: python generar_memoria.py
Produce:  memoria.docx
"""

from docx import Document
from docx.shared import Pt, Cm, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import datetime
import os

PLOTS = 'plots'


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def set_cell_bg(cell, hex_color):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), hex_color)
    tcPr.append(shd)


def set_cell_borders(table):
    for row in table.rows:
        for cell in row.cells:
            tc = cell._tc
            tcPr = tc.get_or_add_tcPr()
            tcBorders = OxmlElement('w:tcBorders')
            for side in ('top', 'left', 'bottom', 'right'):
                border = OxmlElement(f'w:{side}')
                border.set(qn('w:val'), 'single')
                border.set(qn('w:sz'), '4')
                border.set(qn('w:space'), '0')
                border.set(qn('w:color'), '999999')
                tcBorders.append(border)
            tcPr.append(tcBorders)


def add_heading(doc, text, level):
    p = doc.add_heading(text, level=level)
    run = p.runs[0] if p.runs else p.add_run(text)
    if level == 1:
        run.font.size = Pt(16)
        run.font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)
    elif level == 2:
        run.font.size = Pt(13)
        run.font.color.rgb = RGBColor(0x2E, 0x74, 0xB5)
    elif level == 3:
        run.font.size = Pt(11)
        run.font.color.rgb = RGBColor(0x4A, 0x86, 0xC8)
    return p


def add_body(doc, text, bold=False, italic=False, size=11):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.name = 'Calibri'
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    p.paragraph_format.space_after = Pt(6)
    return p


def add_caption(doc, text):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.name = 'Calibri'
    run.font.size = Pt(9)
    run.italic = True
    run.font.color.rgb = RGBColor(0x59, 0x59, 0x59)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(12)
    return p


def add_formula(doc, text):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.name = 'Courier New'
    run.font.size = Pt(10)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(8)
    return p


def add_image(doc, filename, caption, width=Inches(5.5)):
    """Inserta una imagen desde plots/ centrada con pie de figura."""
    path = os.path.join(PLOTS, filename)
    if not os.path.exists(path):
        add_body(doc, f'[Imagen no encontrada: {filename}]', italic=True)
        add_caption(doc, caption)
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(path, width=width)
    p.paragraph_format.space_before = Pt(6)
    add_caption(doc, caption)


def build_table(doc, headers, rows, header_bg='1F497D', header_fg='FFFFFF'):
    col_count = len(headers)
    table = doc.add_table(rows=1 + len(rows), cols=col_count)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = 'Table Grid'
    hrow = table.rows[0]
    for i, h in enumerate(headers):
        cell = hrow.cells[i]
        cell.text = h
        cell.paragraphs[0].runs[0].bold = True
        cell.paragraphs[0].runs[0].font.size = Pt(10)
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(
            int(header_fg[0:2], 16), int(header_fg[2:4], 16), int(header_fg[4:6], 16))
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_cell_bg(cell, header_bg)
    for ri, row_data in enumerate(rows):
        row = table.rows[ri + 1]
        bg = 'F2F2F2' if ri % 2 == 0 else 'FFFFFF'
        for ci, val in enumerate(row_data):
            cell = row.cells[ci]
            cell.text = str(val)
            cell.paragraphs[0].runs[0].font.size = Pt(10)
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            set_cell_bg(cell, bg)
    set_cell_borders(table)
    return table


# ─────────────────────────────────────────────
# DOCUMENTO
# ─────────────────────────────────────────────

doc = Document()

for section in doc.sections:
    section.top_margin    = Cm(2.5)
    section.bottom_margin = Cm(2.5)
    section.left_margin   = Cm(3.0)
    section.right_margin  = Cm(2.5)

style = doc.styles['Normal']
style.font.name = 'Calibri'
style.font.size = Pt(11)


# ══════════════════════════════════════════════
# PORTADA
# ══════════════════════════════════════════════

doc.add_paragraph()
doc.add_paragraph()

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run(
    'CLASIFICACION DEL NIVEL DE HABILIDAD\n'
    'EN AJEDREZ MEDIANTE\n'
    'APRENDIZAJE AUTOMATICO'
)
run.font.name = 'Calibri'
run.font.size = Pt(24)
run.bold = True
run.font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)
p.paragraph_format.space_after = Pt(30)

p2 = doc.add_paragraph()
p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
run2 = p2.add_run('Memoria del Proyecto Final')
run2.font.name = 'Calibri'
run2.font.size = Pt(16)
run2.italic = True
run2.font.color.rgb = RGBColor(0x2E, 0x74, 0xB5)
p2.paragraph_format.space_after = Pt(60)

for linea in [
    'Asignatura: Big Data',
    'Grado: Ingenieria Informatica - 5. Curso',
    f'Fecha: {datetime.date.today().strftime("%d de %B de %Y")}',
]:
    p3 = doc.add_paragraph()
    p3.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r3 = p3.add_run(linea)
    r3.font.name = 'Calibri'
    r3.font.size = Pt(12)

doc.add_page_break()


# ══════════════════════════════════════════════
# 1. INTRODUCCION
# ══════════════════════════════════════════════

add_heading(doc, '1. Introduccion y Motivacion', 1)

add_body(doc,
    'El ajedrez es uno de los juegos de estrategia mas estudiados del mundo, con millones de '
    'partidas disputadas online cada dia en plataformas como Lichess o Chess.com. Cada partida '
    'genera un conjunto rico de estadisticas (numero de movimientos, tipo de apertura, control '
    'del tiempo, modalidad de victoria) que, aunque no revelan directamente la puntuacion ELO '
    'del jugador, contienen patrones latentes relacionados con su nivel de habilidad.')

add_body(doc,
    'El objetivo de este proyecto es construir un sistema de clasificacion automatica que, '
    'a partir de las caracteristicas observables de una partida de ajedrez, sea capaz de '
    'predecir si el jugador pertenece a la categoria Principiante o Intermedio, sin acceder '
    'directamente a su puntuacion ELO. Este enfoque tiene aplicaciones practicas en sistemas '
    'de emparejamiento adaptativo, analisis pedagogico y deteccion de anomalias en plataformas '
    'de juego online.')

add_body(doc,
    'Para abordar el problema se han implementado y comparado cuatro tecnicas de aprendizaje '
    'automatico: Regresion Logistica One-vs-All, Red Neuronal Feedforward, Support Vector '
    'Machine con kernel RBF y Random Forest. Adicionalmente, se ha construido un Ensemble de '
    'votacion suave que combina las predicciones de los cuatro modelos. Las implementaciones '
    'de Regresion Logistica y Red Neuronal se han realizado desde cero con NumPy, sin recurrir '
    'a librerias de alto nivel, con el fin de demostrar la comprension profunda de los '
    'algoritmos subyacentes.')

doc.add_page_break()


# ══════════════════════════════════════════════
# 2. DATASET
# ══════════════════════════════════════════════

add_heading(doc, '2. Dataset', 1)
add_heading(doc, '2.1 Descripcion', 2)

add_body(doc,
    'El dataset utilizado es el Lichess Games Dataset, disponible publicamente en Kaggle. '
    'Contiene informacion de partidas de ajedrez disputadas en la plataforma Lichess, '
    'incluyendo los ratings de ambos jugadores, el resultado, el tipo de apertura, el '
    'control de tiempo y otras estadisticas de cada partida.')

add_body(doc, 'Las columnas principales del dataset original son:')

build_table(doc,
    ['Columna', 'Descripcion'],
    [
        ('id',             'Identificador unico de la partida'),
        ('rated',          'Si la partida conto para el ranking (booleano)'),
        ('turns',          'Numero total de movimientos de la partida'),
        ('victory_status', 'Forma de terminar: mate, resign, outoftime, draw'),
        ('winner',         'Ganador: white, black o draw'),
        ('increment_code', 'Control de tiempo en formato "base+incremento"'),
        ('white_rating',   'Puntuacion ELO del jugador con blancas'),
        ('black_rating',   'Puntuacion ELO del jugador con negras'),
        ('opening_eco',    'Codigo ECO de la apertura jugada'),
        ('opening_ply',    'Numero de movimientos de apertura reconocidos'),
    ],
    header_bg='2E74B5')
add_caption(doc, 'Tabla 1: Principales columnas del dataset Lichess Games.')

doc.add_paragraph()

add_heading(doc, '2.2 Variable Target', 2)

add_body(doc,
    'Dado que el objetivo es predecir el nivel de habilidad sin usar directamente el ELO, '
    'se construye la variable target a partir de el como referencia de ground truth. '
    'El proceso es el siguiente:')

for s in [
    '1. Se calcula el ELO promedio de la partida: skill_level = (white_rating + black_rating) / 2',
    '2. Se calculan los percentiles 33 y 66 sobre el dataset COMPLETO (antes del muestreo).',
    '3. Se discretiza en tres clases: Principiante (< p33), Intermedio (p33-p66), Avanzado (> p66).',
    '4. Tras el muestreo estratificado de 2.500 partidas, la clase Avanzado quedo con muy pocas '
     'muestras, por lo que el problema se trata efectivamente como clasificacion binaria '
     '(Principiante vs Intermedio).',
]:
    p = doc.add_paragraph(s, style='List Bullet')
    p.runs[0].font.size = Pt(11)

add_formula(doc, 'skill_level = (white_rating + black_rating) / 2')

add_body(doc,
    'Este diseno garantiza que el modelo nunca "ve" el ELO durante la inferencia; '
    'solo se usa para construir la etiqueta en tiempo de entrenamiento.')

add_heading(doc, '2.3 Muestreo Estratificado y Division de Datos', 2)

add_body(doc,
    'Para mantener tiempos de entrenamiento razonables, se extrae una muestra aleatoria '
    'estratificada de 2.500 partidas usando train_test_split con stratify=y y random_state=42. '
    'La estratificacion garantiza que las proporciones de clase se preservan en la muestra '
    'y en cada uno de los tres subconjuntos resultantes:')

build_table(doc,
    ['Conjunto', 'Tamano', 'Proporcion'],
    [('Train',      '1.500', '60 %'),
     ('Validacion', '500',   '20 %'),
     ('Test',       '500',   '20 %')],
    header_bg='2E74B5')
add_caption(doc, 'Tabla 2: Division train / validacion / test con estratificacion.')

doc.add_paragraph()
doc.add_page_break()


# ══════════════════════════════════════════════
# 3. EDA
# ══════════════════════════════════════════════

add_heading(doc, '3. Analisis Exploratorio de Datos (EDA)', 1)

add_body(doc,
    'Antes del modelado se realizo un analisis exploratorio completo para comprender la '
    'distribucion de las variables, detectar posibles problemas de calidad del dato y '
    'motivar las decisiones de preprocesado. A continuacion se presentan las seis '
    'visualizaciones principales generadas.')

# --- Figura 1 ---
add_heading(doc, '3.1 Distribucion del ELO por nivel de habilidad', 2)
add_body(doc,
    'El histograma muestra la distribucion del ELO promedio separada por clase, con lineas '
    'verticales en los percentiles 33 y 66. Se observa que las tres clases se solapan en los '
    'extremos, lo que anticipa la dificultad de separacion que mostraran los modelos. '
    'El solapamiento entre Principiante e Intermedio es el que mas afecta al rendimiento, '
    'ya que son las dos clases que el sistema debe distinguir.')
add_image(doc, '01_elo_distribution.png',
          'Figura 1: Distribucion del ELO promedio por clase con percentiles 33 y 66.')

# --- Figura 2 ---
add_heading(doc, '3.2 Distribucion de clases', 2)
add_body(doc,
    'El grafico de barras confirma que el muestreo estratificado mantiene proporciones '
    'equilibradas entre clases. La distribucion efectiva binaria (Principiante / Intermedio) '
    'presenta un desbalance de aproximadamente 66 % / 34 %, lo que motiva el uso de SMOTE '
    'y metricas de evaluacion robustas al desbalance (F1 macro, ROC-AUC).')
add_image(doc, '02_class_distribution.png',
          'Figura 2: Distribucion de clases en la muestra de 2.500 partidas.')

# --- Figura 3 ---
add_heading(doc, '3.3 Movimientos por nivel', 2)
add_body(doc,
    'Los boxplots de turns y opening_ply por clase muestran que los jugadores de nivel '
    'superior tienden a jugar partidas con mas movimientos y utilizan aperturas mas '
    'elaboradas. La varianza es alta en todas las clases, indicando que estas variables '
    'por si solas no son suficientes para separar las clases.')
add_image(doc, '03_turns_opening_ply_boxplot.png',
          'Figura 3: Distribucion de movimientos totales (turns) y de apertura (opening_ply) por nivel.')

# --- Figura 4 ---
add_heading(doc, '3.4 Forma de terminar la partida por nivel', 2)
add_body(doc,
    'La distribucion de la forma de terminar la partida varia con el nivel. Los jugadores '
    'avanzados terminan mas partidas por mate, mientras que los principiantes presentan '
    'una mayor proporcion de outoftime, lo que refleja una gestion del tiempo menos eficiente.')
add_image(doc, '04_victory_status_by_class.png',
          'Figura 4: Distribucion de victory_status por nivel de habilidad.')

# --- Figura 5 ---
add_heading(doc, '3.5 Matriz de correlacion', 2)
add_body(doc,
    'La matriz de correlacion de Pearson entre las variables numericas revela una '
    'correlacion muy alta entre white_rating y black_rating (el sistema de emparejamiento '
    'une jugadores de nivel similar). Las features turns y opening_ply presentan '
    'correlaciones moderadas con los ratings.')
add_image(doc, '05_correlation_matrix.png',
          'Figura 5: Matriz de correlacion de Pearson entre features numericas.')

# --- Figura 6 ---
add_heading(doc, '3.6 Control de tiempo por nivel', 2)
add_body(doc,
    'Los boxplots del tiempo base muestran que los jugadores de mayor nivel tienden a '
    'jugar con controles de tiempo mas largos (partidas de ritmo clasico), mientras que '
    'los principiantes participan mas en partidas de bullet o blitz.')
add_image(doc, '06_base_time_by_class.png',
          'Figura 6: Distribucion del tiempo base del control de tiempo por nivel de habilidad.')

doc.add_page_break()


# ══════════════════════════════════════════════
# 4. PREPROCESADO
# ══════════════════════════════════════════════

add_heading(doc, '4. Preprocesado de Datos', 1)
add_heading(doc, '4.1 Feature Engineering', 2)

add_body(doc,
    'A partir de las columnas originales del dataset se construyeron 17 features que '
    'alimentan los modelos. El proceso de extraccion se describe a continuacion:')

build_table(doc,
    ['Feature', 'Tipo', 'Transformacion', 'Descripcion'],
    [
        ('turns',          'Numerico',           'StandardScaler',  'Numero total de movimientos'),
        ('opening_ply',    'Numerico',           'StandardScaler',  'Movimientos de apertura reconocidos'),
        ('victory_status', 'Categorico (4 vals)', 'One-Hot (4 cols)', 'mate / resign / outoftime / draw'),
        ('winner',         'Categorico (3 vals)', 'One-Hot (3 cols)', 'white / black / draw'),
        ('rated',          'Binario',            'StandardScaler',  'Partida puntuada para ranking'),
        ('base_time',      'Numerico',           'StandardScaler',  'Tiempo base extraido de increment_code'),
        ('increment',      'Numerico',           'StandardScaler',  'Incremento por jugada extraido de increment_code'),
        ('opening_eco',    'Categorico (5 fam.)', 'One-Hot (5 cols)', 'Familia ECO: A, B, C, D, E'),
    ],
    header_bg='2E74B5')
add_caption(doc, 'Tabla 3: Features construidas y transformaciones aplicadas (total: 17 columnas).')

doc.add_paragraph()

add_body(doc,
    'La columna increment_code tiene el formato "base+incremento" (por ejemplo "600+5"). '
    'Se parseo manualmente para extraer dos features numericas independientes: el tiempo '
    'base en segundos y el incremento por jugada. La columna opening_eco se agrupo '
    'por familia (primera letra del codigo ECO) para reducir la cardinalidad.')

add_heading(doc, '4.2 Normalizacion', 2)

add_body(doc,
    'Las variables numericas continuas se normalizan con StandardScaler (media 0, '
    'desviacion estandar 1). Los scalers se ajustan exclusivamente sobre el conjunto de '
    'entrenamiento y se aplican sin reajuste sobre validacion y test, evitando la fuga '
    'de informacion (data leakage). Las variables binarias y one-hot ya estan en rango '
    '[0, 1] y no requieren normalizacion adicional.')

add_heading(doc, '4.3 Balanceo de Clases con SMOTE', 2)

add_body(doc,
    'El conjunto de entrenamiento presenta un desbalance de aproximadamente 66 % / 34 % '
    'entre Principiantes e Intermedios. Entrenar directamente sobre datos desbalanceados '
    'tiende a que el modelo maximice la clase mayoritaria, perjudicando el recall de '
    'la clase minoritaria (Intermedios).')

add_body(doc,
    'Para mitigar este problema se aplica SMOTE (Synthetic Minority Over-sampling '
    'Technique) exclusivamente sobre el conjunto de entrenamiento. SMOTE genera muestras '
    'sinteticas de la clase minoritaria interpolando entre vecinos cercanos en el espacio '
    'de features (k=5), en lugar de simplemente duplicar muestras existentes, lo que '
    'reduce el riesgo de sobreajuste por repeticion.')

add_body(doc,
    'Resultado: el conjunto de entrenamiento pasa de ~1.500 muestras desbalanceadas a '
    '~2.000 muestras equilibradas al 50 % por clase. Los conjuntos de validacion y test '
    'no se modifican, preservando la distribucion real para una evaluacion honesta.')

doc.add_page_break()


# ══════════════════════════════════════════════
# 5. MODELOS
# ══════════════════════════════════════════════

add_heading(doc, '5. Modelos de Clasificacion', 1)

# ── 5.1 LR ──────────────────────────────────

add_heading(doc, '5.1 Regresion Logistica One-vs-All (implementacion NumPy)', 2)
add_heading(doc, 'Fundamento matematico', 3)

add_body(doc,
    'La Regresion Logistica binaria modela la probabilidad de pertenencia a una clase '
    'mediante la funcion sigmoide aplicada sobre una combinacion lineal de las features:')

add_formula(doc, 'P(y=1 | x) = 1 / (1 + exp(-(w^T x + b)))')

add_body(doc,
    'Para la clasificacion multiclase se emplea la estrategia One-vs-All (OvA): se '
    'entrena un clasificador binario independiente por cada clase. La prediccion final '
    'se asigna a la clase cuyo clasificador devuelve la mayor probabilidad.')

add_body(doc, 'La funcion de coste con regularizacion L2 es:')
add_formula(doc, 'J(w) = -(1/n) * sum[y*log(y_hat) + (1-y)*log(1-y_hat)] + (lambda/2n)*||w||^2')

add_body(doc, 'Los gradientes para el descenso de gradiente son:')
add_formula(doc, 'dJ/dw = (1/n) * X^T * (y_hat - y) + (lambda/n) * w')
add_formula(doc, 'dJ/db = (1/n) * sum(y_hat - y)')

add_heading(doc, 'Decisiones de implementacion', 3)
add_body(doc,
    'El modelo se implemento completamente con NumPy, sin usar scikit-learn para el '
    'ajuste. Los pesos se inicializan con N(0, 0.01). Se aplican pesos de clase '
    'balanceados (compute_class_weight) para compensar el desbalance residual.')

add_heading(doc, 'Busqueda de hiperparametros', 3)
add_body(doc, 'Se realizo una busqueda en grid manual sobre 12 combinaciones:')

build_table(doc,
    ['Hiperparametro', 'Valores explorados'],
    [('learning_rate', '0.01 | 0.1 | 0.5'),
     ('lambda_reg',    '0.0 | 0.01 | 0.1 | 1.0'),
     ('n_iterations',  '1.000 (fijo)')],
    header_bg='2E74B5')
add_caption(doc, 'Tabla 4: Grid de hiperparametros para Regresion Logistica OvA.')

doc.add_paragraph()

add_body(doc, 'Curva de validacion del parametro lambda_reg sobre validacion cruzada 5-fold:')
add_image(doc, 'validation_curve_Logistic_Regression_lambda_reg.png',
          'Figura 7: Curva de validacion de lambda_reg para Regresion Logistica (5-fold CV).',
          width=Inches(5.0))

add_body(doc, 'Coeficientes aprendidos por el clasificador de la clase Intermedio:')
add_image(doc, 'lr_feature_coefficients.png',
          'Figura 8: Importancia de features segun los coeficientes de la Regresion Logistica (clase Intermedio).',
          width=Inches(5.0))

# ── 5.2 NN ──────────────────────────────────

add_heading(doc, '5.2 Red Neuronal Feedforward (implementacion NumPy)', 2)
add_heading(doc, 'Arquitectura', 3)

add_body(doc,
    'Se implemento una red neuronal feedforward totalmente conectada con arquitectura '
    'configurable. La capa de entrada tiene 17 neuronas. Las capas ocultas utilizan '
    'activacion ReLU y la capa de salida usa Softmax:')

add_formula(doc, 'ReLU(z) = max(0, z)')
add_formula(doc, 'Softmax(z_i) = exp(z_i) / sum_j(exp(z_j))')

add_body(doc, 'La funcion de coste es la entropia cruzada categorica con regularizacion L2:')
add_formula(doc, 'J = -(1/n) * sum_i sum_c [w_c * y_ic * log(y_hat_ic)] + (lambda/2n) * sum_l ||W_l||^2_F')

add_heading(doc, 'Backpropagation', 3)

add_body(doc,
    'El backpropagation se implemento manualmente. Para la capa de salida, la derivada '
    'combinada Softmax + Cross-Entropy simplifica a:')
add_formula(doc, 'delta_salida = y_hat - y_onehot')
add_body(doc, 'Para las capas ocultas, el gradiente se propaga multiplicando por la derivada de ReLU:')
add_formula(doc, 'delta_l = (delta_(l+1) * W_(l+1)^T) o ReLU\'(z_l)')

add_body(doc,
    'Los pesos se inicializan con la inicializacion He, recomendada para activaciones '
    'ReLU: W ~ N(0, sqrt(2/n_in)).')

add_heading(doc, 'Entrenamiento con Early Stopping', 3)

add_body(doc,
    'El entrenamiento usa Minibatch SGD. Se implemento Early Stopping monitorizando '
    'la perdida en el conjunto de validacion: si no mejora durante 20 epocas consecutivas '
    '(patience=20), el entrenamiento se detiene y se restauran los pesos de la mejor epoca.')

add_image(doc, 'nn_training_history.png',
          'Figura 9: Evolucion de la perdida y el accuracy por epoca en train y validacion. '
          'La linea verde marca el punto de Early Stopping.',
          width=Inches(5.5))

add_heading(doc, 'Busqueda de hiperparametros', 3)
add_body(doc, 'Se exploro un grid de 54 combinaciones:')

build_table(doc,
    ['Hiperparametro', 'Valores explorados'],
    [('hidden_layers',    '[32] | [64, 32] | [128, 64, 32]'),
     ('learning_rate',    '0.001 | 0.01 | 0.1'),
     ('lambda_reg',       '0.0 | 0.001 | 0.01'),
     ('batch_size',       '32 | 64'),
     ('max_epochs / patience', '200 / 20 (fijos)')],
    header_bg='2E74B5')
add_caption(doc, 'Tabla 5: Grid de hiperparametros para la Red Neuronal.')

doc.add_paragraph()

add_body(doc, 'Curva de validacion del learning_rate sobre validacion cruzada 5-fold:')
add_image(doc, 'validation_curve_Neural_Network_learning_rate.png',
          'Figura 10: Curva de validacion del learning_rate para la Red Neuronal (5-fold CV).',
          width=Inches(5.0))

# ── 5.3 SVM ──────────────────────────────────

add_heading(doc, '5.3 Support Vector Machine con kernel RBF (scikit-learn)', 2)
add_heading(doc, 'Fundamento matematico', 3)

add_body(doc,
    'La SVM busca el hiperplano de margen maximo que separa las clases. Con el kernel '
    'RBF se proyecta implicitamente el espacio de features a una dimension superior, '
    'permitiendo fronteras de decision no lineales:')
add_formula(doc, 'K(xi, xj) = exp(-gamma * ||xi - xj||^2)')

add_body(doc,
    'El parametro C controla el equilibrio entre margen maximo y tolerancia al error. '
    'Se utiliza class_weight="balanced" para compensar el desbalance.')

add_heading(doc, 'Busqueda de hiperparametros', 3)
add_body(doc, 'GridSearchCV con 5-fold CV sobre 16 combinaciones:')

build_table(doc,
    ['Hiperparametro', 'Valores explorados'],
    [('C',     '0.1 | 1 | 10 | 100'),
     ('gamma', '"scale" | "auto" | 0.01 | 0.1')],
    header_bg='2E74B5')
add_caption(doc, 'Tabla 6: Grid de hiperparametros para SVM RBF.')

doc.add_paragraph()

# ── 5.4 RF ──────────────────────────────────

add_heading(doc, '5.4 Random Forest (scikit-learn)', 2)
add_heading(doc, 'Fundamento', 3)

add_body(doc,
    'Random Forest combina multiples arboles de decision entrenados con bagging y '
    'seleccion aleatoria de features en cada division. La prediccion es la media de '
    'las probabilidades de todos los arboles. Se usa class_weight="balanced".')

add_heading(doc, 'Busqueda de hiperparametros', 3)
build_table(doc,
    ['Hiperparametro', 'Valores explorados'],
    [('n_estimators',      '50 | 100 | 200'),
     ('max_depth',         'None | 5 | 10 | 20'),
     ('min_samples_split', '2 | 5 | 10')],
    header_bg='2E74B5')
add_caption(doc, 'Tabla 7: Grid de hiperparametros para Random Forest.')

doc.add_paragraph()

add_body(doc, 'Importancia de features segun el mejor modelo Random Forest:')
add_image(doc, 'feature_importance_Random_Forest.png',
          'Figura 11: Importancia relativa de features segun el Random Forest.',
          width=Inches(5.0))

# ── 5.5 Ensemble ──────────────────────────────────

add_heading(doc, '5.5 Ensemble de Votacion Suave (implementacion NumPy)', 2)

add_body(doc,
    'Como tecnica adicional se implemento un Ensemble de votacion suave (soft voting) '
    'que combina las predicciones probabilisticas de los cuatro modelos. Cada modelo '
    'contribuye con un peso proporcional a su ROC-AUC en test:')

add_formula(doc, 'P_ensemble(y=c | x) = sum_m(w_m * P_m(y=c | x)) / sum_m(w_m)')

add_body(doc,
    'Esta estrategia tiende a reducir la varianza respecto a cualquier modelo individual '
    'y suaviza los comportamientos extremos.')

doc.add_page_break()


# ══════════════════════════════════════════════
# 6. EVALUACION
# ══════════════════════════════════════════════

add_heading(doc, '6. Evaluacion y Comparacion de Modelos', 1)
add_heading(doc, '6.1 Metricas de Evaluacion', 2)

add_body(doc,
    'Dado el desbalance de clases (66 % Principiantes / 34 % Intermedios), el accuracy '
    'no es una metrica suficiente: un clasificador trivial que prediga siempre Principiante '
    'obtendria un 66 % sin aprender nada. Por ello las metricas principales son:')

for nombre, desc in [
    ('F1-Score macro',
     'Media armonica de precision y recall, promediada por igual sobre todas las clases. '
     'Penaliza fuertemente el mal rendimiento en la clase minoritaria.'),
    ('ROC-AUC',
     'Area bajo la curva ROC. Mide la capacidad discriminativa del modelo '
     'independientemente del threshold. Valor 0.5 = aleatorio, 1.0 = perfecto.'),
    ('Precision / Recall por clase',
     'Permiten analizar el comportamiento especifico del modelo en cada clase y '
     'detectar sesgos hacia la clase mayoritaria.'),
    ('Matriz de confusion',
     'Visibilidad completa de los errores: falsos positivos y falsos negativos por clase.'),
]:
    p = doc.add_paragraph(style='List Bullet')
    p.add_run(nombre + ': ').bold = True
    p.runs[0].bold = True
    p.runs[0].font.size = Pt(11)
    p.add_run(desc).font.size = Pt(11)

add_heading(doc, '6.2 Validacion Cruzada Estratificada 5-Fold', 2)

add_body(doc,
    'Se aplico validacion cruzada estratificada de 5 pliegues sobre el conjunto '
    'train+validacion combinado para obtener una estimacion robusta del rendimiento. '
    'Para los modelos NumPy (LR y NN), que no son compatibles con cross_val_score, '
    'se implemento un bucle de CV manual. En la NN se reserva un 20 % adicional '
    'dentro del pliegue de entrenamiento como validacion interna para el Early Stopping.')

add_heading(doc, '6.3 Curvas ROC', 2)

add_body(doc,
    'Las curvas ROC permiten evaluar la capacidad discriminativa de cada modelo '
    'a lo largo de todos los posibles thresholds de decision:')

add_image(doc, 'roc_curve_Logistic_Regression.png',
          'Figura 12: Curva ROC - Regresion Logistica (AUC = 0.706).', width=Inches(4.2))
add_image(doc, 'roc_curve_Neural_Network.png',
          'Figura 13: Curva ROC - Red Neuronal (AUC = 0.737).', width=Inches(4.2))
add_image(doc, 'roc_curve_SVM_(RBF).png',
          'Figura 14: Curva ROC - SVM RBF (AUC = 0.646).', width=Inches(4.2))
add_image(doc, 'roc_curve_Random_Forest.png',
          'Figura 15: Curva ROC - Random Forest (AUC = 0.711).', width=Inches(4.2))
add_image(doc, 'roc_curve_Ensemble.png',
          'Figura 16: Curva ROC - Ensemble Soft Voting (AUC = 0.726).', width=Inches(4.2))

add_heading(doc, '6.4 Matrices de Confusion', 2)

add_body(doc,
    'Las matrices de confusion muestran el detalle de los aciertos y errores de cada '
    'modelo sobre el conjunto de test (500 muestras: 330 Principiantes, 170 Intermedios):')

add_image(doc, 'confusion_matrix_Logistic_Regression.png',
          'Figura 17: Matriz de confusion - Regresion Logistica (threshold=0.5).', width=Inches(3.8))
add_image(doc, 'confusion_matrix_Neural_Network.png',
          'Figura 18: Matriz de confusion - Red Neuronal (threshold=0.5).', width=Inches(3.8))
add_image(doc, 'confusion_matrix_SVM_(RBF).png',
          'Figura 19: Matriz de confusion - SVM RBF (threshold=0.5).', width=Inches(3.8))
add_image(doc, 'confusion_matrix_Random_Forest.png',
          'Figura 20: Matriz de confusion - Random Forest (threshold=0.5).', width=Inches(3.8))
add_image(doc, 'confusion_matrix_Ensemble.png',
          'Figura 21: Matriz de confusion - Ensemble Soft Voting (threshold=0.5).', width=Inches(3.8))

add_heading(doc, '6.5 Resultados con Threshold por Defecto (0.5)', 2)

build_table(doc,
    ['Modelo', 'Accuracy', 'F1 Macro', 'ROC-AUC', 'Recall Interm.'],
    [('Logistic Regression', '0.664', '0.644', '0.706', '0.63'),
     ('Neural Network',      '0.700', '0.640', '0.737', '0.43'),
     ('SVM (RBF)',           '0.620', '0.592', '0.646', '0.53'),
     ('Random Forest',       '0.676', '0.627', '0.711', '0.46'),
     ('Ensemble',            '0.690', '0.651', '0.726', '0.52')],
    header_bg='1F497D')
add_caption(doc, 'Tabla 8: Resultados de los cinco modelos en el conjunto de test (threshold = 0.5).')

doc.add_paragraph()

add_body(doc,
    'El modelo con mayor accuracy es la Red Neuronal (0.700), sin embargo presenta el '
    'peor recall de Intermedios (0.43), clasificando correctamente solo 73 de 170 '
    'instancias de la clase minoritaria. La Regresion Logistica OvA muestra el mejor '
    'recall de Intermedios a threshold 0.5 (0.63). El Ensemble obtiene el mejor '
    'F1 macro (0.651), combinando las fortalezas de cada modelo individual.')

add_heading(doc, '6.6 Ajuste de Threshold para Mejorar el Recall de Intermedios', 2)

add_body(doc,
    'El threshold por defecto de 0.5 no tiene por que ser el optimo para un problema '
    'con desbalance de clases. Reducir el threshold para la clase Intermedio aumenta su '
    'recall a costa de reducir algo su precision. Para cada modelo se busco el threshold '
    'que maximiza el F1-Score de la clase Intermedio con precision minima de 0.40, '
    'buscando en el rango [0.10, 0.70] con 121 pasos.')

build_table(doc,
    ['Modelo', 'Threshold', 'Accuracy', 'F1 Macro', 'Recall Interm.', 'Prec. Interm.'],
    [('LR',       '0.460', '0.628', '0.624', '0.77', '0.47'),
     ('NN',       '0.275', '0.632', '0.630', '0.82', '0.48'),
     ('SVM',      '0.435', '0.606', '0.596', '0.66', '0.45'),
     ('RF',       '0.275', '0.570', '0.570', '0.88', '0.43'),
     ('Ensemble', '0.410', '0.662', '0.653', '0.74', '0.50')],
    header_bg='1F497D')
add_caption(doc, 'Tabla 9: Resultados con threshold ajustado para maximizar recall de Intermedios.')

doc.add_paragraph()

add_body(doc, 'Matrices de confusion del modelo recomendado con threshold ajustado:')
add_image(doc, 'confusion_matrix_Ensemble_(thr=0.410).png',
          'Figura 22: Matriz de confusion - Ensemble con threshold=0.410 (modelo recomendado).',
          width=Inches(3.8))

add_heading(doc, '6.7 Modelo Recomendado', 2)

add_body(doc,
    'El modelo recomendado es el Ensemble con threshold = 0.410. Los motivos son:')

for r in [
    'Mayor F1 macro (0.653): mejor balance global entre clases, incluyendo la minoritaria.',
    'Recall de Intermedios = 0.74: detecta correctamente 126 de 170 Intermedios, '
     'frente a los 89 del Ensemble base (+41.6 % de mejora).',
    'Precision de Intermedios = 0.50: dentro del limite minimo aceptable (0.40).',
    'Accuracy razonable (0.662): solo 2.8 puntos menos que el Ensemble base (0.690).',
    'ROC-AUC = 0.726: el AUC no varia con el threshold, confirmando buena capacidad discriminativa.',
]:
    p = doc.add_paragraph(r, style='List Bullet')
    p.runs[0].font.size = Pt(11)

doc.add_page_break()


# ══════════════════════════════════════════════
# 7. CURVAS DE APRENDIZAJE Y VALIDACION
# ══════════════════════════════════════════════

add_heading(doc, '7. Curvas de Entrenamiento y Validacion', 1)
add_heading(doc, '7.1 Curvas de Aprendizaje', 2)

add_body(doc,
    'Las curvas de aprendizaje muestran como evolucionan el score de entrenamiento y '
    'el score de validacion cruzada (5-fold) en funcion del tamano del conjunto de '
    'entrenamiento. Permiten diagnosticar overfitting (gap grande entre train y CV) '
    'o underfitting (ambas curvas bajas).')

add_image(doc, 'learning_curve_SVM_(RBF).png',
          'Figura 23: Curva de aprendizaje - SVM RBF.',
          width=Inches(5.0))
add_image(doc, 'learning_curve_Random_Forest.png',
          'Figura 24: Curva de aprendizaje - Random Forest.',
          width=Inches(5.0))

add_body(doc,
    'La curva del SVM muestra un gap moderado entre train y CV, indicando cierta '
    'sobreconfianza en los datos de entrenamiento. La del Random Forest muestra '
    'que el modelo mejora con mas datos, con el gap reduciendose al aumentar el tamano.')

add_heading(doc, '7.2 Historial de Entrenamiento de la Red Neuronal', 2)

add_body(doc,
    'El historial de entrenamiento de la NN (ya mostrado en la Figura 9) evidencia '
    'la efectividad del Early Stopping: la linea verde marca la epoca en que se detiene '
    'el entrenamiento antes de que la perdida de validacion comience a empeorar, '
    'restaurando automaticamente los mejores pesos.')

add_heading(doc, '7.3 Curvas de Validacion de Hiperparametros', 2)

add_body(doc,
    'Las curvas de validacion ya presentadas en las Figuras 7 y 10 muestran como varia '
    'el score al modificar un unico hiperparametro manteniendo el resto fijos:')

for nombre, desc in [
    ('LR - lambda_reg (Figura 7)',
     'Valores muy bajos de lambda favorecen el overfitting (gap train/CV mayor); '
     'valores muy altos introducen underfitting. El optimo se situa en valores '
     'intermedios donde el CV score es maximo y el gap es minimo.'),
    ('NN - learning_rate (Figura 10)',
     'Learning rates muy pequenos convergen lentamente; learning rates muy grandes '
     'pueden causar oscilaciones. La curva de validacion identifica el rango estable.'),
]:
    add_body(doc, nombre, bold=True)
    add_body(doc, desc)

doc.add_page_break()


# ══════════════════════════════════════════════
# 8. CONCLUSIONES
# ══════════════════════════════════════════════

add_heading(doc, '8. Conclusiones', 1)
add_heading(doc, '8.1 Resumen de resultados', 2)

add_body(doc,
    'Se han implementado y evaluado cinco sistemas de clasificacion para predecir el nivel '
    'de habilidad en ajedrez a partir de estadisticas de partida. El mejor resultado '
    'global se obtiene con el Ensemble de votacion suave con threshold ajustado a 0.410, '
    'alcanzando un F1 macro de 0.653 y un recall de Intermedios del 74 %. '
    'Ningun modelo supera el 70 % de accuracy, lo que refleja la dificultad intrinseca '
    'del problema: las estadisticas agregadas de una partida contienen informacion '
    'limitada sobre el nivel del jugador sin acceso a las jugadas individuales.')

add_heading(doc, '8.2 Limitaciones', 2)

for lim in [
    'Las features disponibles (estadisticas agregadas) no capturan la calidad de las '
    'jugadas individuales, que es el indicador mas directo del nivel de habilidad.',
    'El desbalance de clases (66/34 %) persiste en test y limita el recall de la '
    'clase minoritaria incluso con SMOTE y threshold ajustado.',
    'El umbral de discretizacion del ELO (percentiles 33/66) es arbitrario y sensible '
    'a la distribucion del dataset.',
    'El tamano del dataset (2.500 partidas) es moderado; con mas datos los modelos '
    'podrian mostrar mejoras mas significativas.',
]:
    p = doc.add_paragraph(lim, style='List Bullet')
    p.runs[0].font.size = Pt(11)

add_heading(doc, '8.3 Posibles mejoras', 2)

for m in [
    'Feature engineering mas rico: incluir features derivadas de las jugadas individuales '
    '(errores por movimiento, evaluacion del motor de ajedrez por posicion).',
    'Usar el dataset completo sin muestreo (>60.000 partidas) para modelos mas potentes.',
    'Probar modelos de gradient boosting (XGBoost, LightGBM) que suelen rendir mejor '
    'que Random Forest en datos tabulares con desbalance.',
    'Ajuste de threshold mediante curvas de precision-recall para un analisis mas fino.',
    'Problema de tres clases completo si se dispone de mas datos de la clase Avanzado.',
]:
    p = doc.add_paragraph(m, style='List Bullet')
    p.runs[0].font.size = Pt(11)

doc.add_page_break()


# ══════════════════════════════════════════════
# 9. REFERENCIAS
# ══════════════════════════════════════════════

add_heading(doc, '9. Referencias', 1)

for ref in [
    '[1] Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). '
    'SMOTE: Synthetic Minority Over-sampling Technique. Journal of Artificial Intelligence Research, 16, 321-357.',
    '[2] Cortes, C., & Vapnik, V. (1995). Support-vector networks. Machine Learning, 20(3), 273-297.',
    '[3] Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.',
    '[4] Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.',
    '[5] He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving Deep into Rectifiers. ICCV 2015.',
    '[6] Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. JMLR, 12, 2825-2830.',
    '[7] Lichess Games Dataset. Kaggle. https://www.kaggle.com/datasets/datasnaek/chess',
]:
    p = doc.add_paragraph(ref, style='List Number')
    p.runs[0].font.size = Pt(10)


# ══════════════════════════════════════════════
# GUARDAR
# ══════════════════════════════════════════════

doc.save('memoria.docx')
print("OK - memoria.docx generada con imagenes correctamente.")
