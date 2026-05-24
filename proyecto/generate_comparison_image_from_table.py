import os

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np


def main():
    plots_dir = 'plots'
    os.makedirs(plots_dir, exist_ok=True)

    models = [
        'Logistic Regression',
        'Neural Network',
        'SVM (RBF)',
        'Random Forest',
        'Ensemble',
    ]

    metrics = {
        'Accuracy': [0.664, 0.700, 0.620, 0.676, 0.690],
        'F1 Macro': [0.644, 0.640, 0.592, 0.627, 0.651],
        'ROC-AUC': [0.706, 0.737, 0.646, 0.711, 0.726],
        'Recall Intermedio': [0.630, 0.430, 0.530, 0.460, 0.520],
    }

    colors = {
        'Accuracy': '#2E86AB',
        'F1 Macro': '#3AAFA9',
        'ROC-AUC': '#F4A261',
        'Recall Intermedio': '#E76F51',
    }

    plt.style.use('ggplot')
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=True)
    axes = axes.ravel()
    y = np.arange(len(models))

    for ax, (metric_name, values) in zip(axes, metrics.items()):
        values = np.array(values, dtype=float)
        bars = ax.barh(y, values, color=colors[metric_name], height=0.62)
        ax.set_title(metric_name, fontsize=13, fontweight='bold')
        ax.set_xlim(0, 0.8)
        ax.set_xticks(np.arange(0, 0.81, 0.1))
        ax.grid(True, axis='x', alpha=0.28)

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_width() + 0.012,
                bar.get_y() + bar.get_height() / 2,
                f'{value:.3f}',
                va='center',
                ha='left',
                fontsize=10,
                color='#333333',
            )

            ax.invert_yaxis()

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(models)
    axes[2].set_yticks(y)
    axes[2].set_yticklabels(models)

    for ax in (axes[1], axes[3]):
        ax.tick_params(axis='y', left=False, labelleft=False)

    fig.subplots_adjust(top=0.88, bottom=0.11, left=0.12, right=0.98, hspace=0.22, wspace=0.07)

    fig.suptitle(
        'Tabla 8 - Resultados de los cinco modelos en el conjunto de prueba (threshold = 0.5)',
        fontsize=15,
        fontweight='bold',
    )
    fig.text(
        0.5,
        0.03,
        'Imagen generada a partir de la tabla aportada por el usuario. No se recalculan modelos ni se usa una nueva semilla.',
        ha='center',
        fontsize=10,
        color='#555555',
    )

    out_path = os.path.join(plots_dir, 'models_comparison_improved.png')
    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f'Imagen guardada en: {out_path}')


if __name__ == '__main__':
    main()