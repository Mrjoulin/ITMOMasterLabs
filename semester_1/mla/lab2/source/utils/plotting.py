import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from selection.compactness import CompactnessProfile

IMAGES_DIR = "./images/"


def _save_and_close(img_name: str):
    if not os.path.exists(IMAGES_DIR):
        os.mkdir(IMAGES_DIR)
        print(f"Created directory for images: {os.path.abspath(IMAGES_DIR)}")

    img_path = os.path.join(IMAGES_DIR, img_name)
    plt.savefig(img_path)
    plt.close('all')


def plot_llo_graphs(k_range, accuracies):
    # Plot dependence of accuracy on k
    plt.figure(figsize=(12, 5))

    best_k = k_range[np.argmax(accuracies)]

    plt.subplot(1, 2, 1)
    plt.plot(k_range, accuracies, marker='o', linestyle='-', color='b', linewidth=2, markersize=6)
    plt.axvline(x=best_k, color='r', linestyle='--', alpha=0.7, label=f'Лучшее k = {best_k}')
    plt.xticks(k_range)
    plt.xlabel('Значение k')
    plt.ylabel('Точность LOO')
    plt.title('Зависимость точности от значения k\n(метод LOO)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    accuracy_changes = np.diff(accuracies) * 100
    plt.bar(k_range[1:], accuracy_changes, color='orange', alpha=0.7)
    plt.xticks(k_range)
    plt.xlabel('Значение k')
    plt.ylabel('Изменение точности (%)')
    plt.title('Изменение точности при увеличении k')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_and_close("llo_graphs.png")


def plot_compactness_full(X, y):
    profile_calc = CompactnessProfile()
    profile_calc.fit(X, y)

    # Computing the full profile
    max_m = min(20, len(X) - 1)
    profile = profile_calc.compute_profile(max_m=max_m)

    plt.figure(figsize=(15, 5))

    xrange = np.arange(1, max_m + 1)
    x_ticks_step = max_m // 10

    plt.subplot(1, 3, 1)
    plt.plot(xrange, profile, 'b-o', linewidth=2, markersize=4)
    plt.xticks(xrange[1::x_ticks_step])
    plt.xlabel('m (порядок соседа)')
    plt.ylabel('Π(m)')
    plt.title('Профиль компактности')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    cumulative = np.cumsum(profile)
    plt.plot(xrange, cumulative, 'r-o', linewidth=2, markersize=4)
    plt.xticks(xrange[1::x_ticks_step])
    plt.xlabel('m (порядок соседа)')
    plt.ylabel('ΣΠ(m)')
    plt.title('Кумулятивный профиль компактности')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    if len(profile) > 1:
        derivative = np.diff(profile)
        plt.plot(xrange[1:], derivative, 'g-o', linewidth=2, markersize=4)
        plt.xticks(xrange[1::x_ticks_step])
        plt.xlabel('m (порядок соседа)')
        plt.ylabel('ΔΠ(m)')
        plt.title('Изменение профиля компактности')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_and_close("compactness_profile.png")

    return profile


def plot_prototype_selection_process(history):
    plt.figure(figsize=(6, 4))

    plt.plot(history["omega_sizes"], history["ccv"], 'b-o', linewidth=2, markersize=4)
    plt.xlabel('Размер множества эталонов |Ω|')
    plt.ylabel('CCV(Ω)')
    plt.title('Изменение CCV в процессе отбора')
    plt.grid(True, alpha=0.3)

    # Mark the minimum point
    min_idx = np.argmin(history["ccv"])
    plt.plot(history["omega_sizes"][min_idx], history["ccv"][min_idx], 'ro', markersize=8)
    plt.tight_layout()
    _save_and_close("prototype_selection_process.png")


def visualize_prototype_selection(X_train, y_train, X_prototypes, y_prototypes):
    """Визуализация исходных данных и отобранных эталонов"""

    # Use PCA to reduce dimensionality to 2D
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pca = PCA(n_components=2)
        X_train_2d = pca.fit_transform(X_train)
        X_prototypes_2d = pca.transform(X_prototypes)

    plt.figure(figsize=(12, 5))

    # Initial data
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(
        X_train_2d[:, 0], X_train_2d[:, 1], c=y_train,
        marker='o', cmap='viridis', alpha=0.6
    )
    plt.colorbar(scatter)
    plt.title('Исходная обучающая выборка')
    xticks, _ = plt.xticks()
    yticks, _ = plt.yticks()
    xlim, ylim = plt.xlim(), plt.ylim()
    plt.xlabel('Главная компонента 1')
    plt.ylabel('Главная компонента 2')

    # Selected prototypes
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(
        X_prototypes_2d[:, 0], X_prototypes_2d[:, 1], c=y_prototypes,
        marker='^', cmap='viridis', s=100, alpha=1
    )
    plt.colorbar(scatter)
    plt.title('Отобранные эталоны')
    plt.xlabel('Главная компонента 1')
    plt.ylabel('Главная компонента 2')
    # Set same scale (ticks and limits)
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.xlim(xlim)
    plt.ylim(ylim)

    plt.tight_layout()
    _save_and_close("prototype_selection_pca.png")
