import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

from .metrics import confusion_matrix

IMAGES_DIR = "./images/"


def _save_and_close(img_name: str):
    if not os.path.exists(IMAGES_DIR):
        os.mkdir(IMAGES_DIR)
        print(f"Created directory for images: {os.path.abspath(IMAGES_DIR)}")

    img_path = os.path.join(IMAGES_DIR, img_name)
    plt.savefig(img_path)
    plt.close('all')


def plot_confusion_matrix(y_true, predictions):
    rows = len(predictions[list(predictions)[0]])
    plt.figure(figsize=(8, 4 * rows))

    for j, method in enumerate(predictions, 1):
        for i, kernel in enumerate(predictions[method]):
            plt.subplot(rows, 2, i * 2 + j)

            preds = predictions[method][kernel]
            conf_mat = confusion_matrix(y_true, preds)
            sns.heatmap(conf_mat.to_numpy(), annot=True, fmt='d', cmap='Blues')
            plt.title(f'Матрица ошибок\n{method} {kernel}')
            plt.xlabel('Предсказанный класс')
            plt.ylabel('Истинный класс')

    plt.tight_layout()
    _save_and_close("confusion_matrix.png")


def visualize_predictions(X, y, predictions):
    """Визуализация исходных данных и отобранных эталонов"""

    # Use PCA to reduce dimensionality to 2D
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)

    rows = len(predictions[list(predictions)[0]])
    plt.figure(figsize=(20, 5 * rows))

    for i in range(rows):
        plt.subplot(rows, 3, i * 3 + 1)
        scatter = plt.scatter(
            X_2d[:, 0], X_2d[:, 1], c=y,
            marker='o', cmap='viridis', alpha=0.6
        )
        plt.colorbar(scatter)
        plt.title('Исходная обучающая выборка')
        plt.xlabel('Главная компонента 1')
        plt.ylabel('Главная компонента 2')

    for j, method in enumerate(predictions, 2):
        for i, kernel in enumerate(predictions[method]):
            model_name = f"{method} {kernel}"
            # Selected prototypes
            plt.subplot(rows, 3, i * 3 + j)
            scatter = plt.scatter(
                X_2d[:, 0], X_2d[:, 1],
                c=predictions[method][kernel],
                marker='o', cmap='viridis', alpha=0.6
            )
            plt.colorbar(scatter)
            plt.title(f'{model_name}')
            plt.xlabel('Главная компонента 1')
            plt.ylabel('Главная компонента 2')

    plt.tight_layout()
    _save_and_close(f"visualize_pca.png")
