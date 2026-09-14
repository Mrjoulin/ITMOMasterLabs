import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from models import CustomPCA

IMAGES_DIR = "./images/"


def _save_and_close(img_name: str):
    if not os.path.exists(IMAGES_DIR):
        os.mkdir(IMAGES_DIR)
        print(f"Created directory for images: {os.path.abspath(IMAGES_DIR)}")

    img_path = os.path.join(IMAGES_DIR, img_name)
    plt.savefig(img_path)
    plt.close('all')


def plot_variance(pca_custom, pca_sklearn):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    n_comp = len(pca_custom.explained_variance_ratio_)

    # Объясненная дисперсия
    axes[0].bar(range(1, n_comp + 1), pca_custom.explained_variance_ratio_,
                alpha=0.7, label='Наша реализация')
    axes[0].bar(range(1, n_comp + 1), pca_sklearn.explained_variance_ratio_,
                alpha=0.7, label='Sklearn', width=0.4)
    axes[0].set_xlabel('Компонента')
    axes[0].set_ylabel('Доля объясненной дисперсии')
    axes[0].set_title('Объясненная дисперсия')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Кумулятивная дисперсия
    axes[1].plot(range(1, n_comp + 1), np.cumsum(pca_custom.explained_variance_ratio_),
                 'b-', marker='o', label='Наша реализация')
    axes[1].plot(range(1, n_comp + 1), np.cumsum(pca_sklearn.explained_variance_ratio_),
                 'r--', marker='s', label='Sklearn')
    axes[1].set_xlabel('Количество компонент')
    axes[1].set_ylabel('Кумулятивная объясненная дисперсия')
    axes[1].set_title('Кумулятивная объясненная дисперсия')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    _save_and_close("variance_compare.png")


def visualize_pca_space(X, y):
    """
    Визуализация данных в пространстве главных компонент
    """

    models = {
        "Наша модель": CustomPCA(n_components=2),
        "Sklearn": PCA(n_components=2)
    }
    plt.figure(figsize=(12, 6))
    for i, (name, model) in enumerate(models.items()):
        plt.subplot(1, 2, i + 1)
        # Применяем PCA
        X_pca = model.fit_transform(X)

        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7, s=50)
        plt.xlabel('Первая главная компонента (PC1)')
        plt.ylabel('Вторая главная компонента (PC2)')
        plt.title(name)
        plt.colorbar(label='Цифра')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_and_close("pca_visualizations.png")
