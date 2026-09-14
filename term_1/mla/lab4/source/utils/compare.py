import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from models import CustomPCA

pd.set_option('display.max_columns', 10)
pd.set_option('display.expand_frame_repr', False)


def compare_with_sklearn(X, n_components=None):
    """
    Сравнение нашей реализации PCA с реализацией из sklearn

    Параметры:
    X: данные
    n_components: количество главных компонент
    """
    # Наша реализация
    pca_custom = CustomPCA(n_components=n_components)
    X_transformed_custom = pca_custom.fit_transform(X)

    # Реализация sklearn
    pca_sklearn = PCA(n_components=n_components, random_state=42)
    X_transformed_sklearn = pca_sklearn.fit_transform(X)

    # Сравнение объясненной дисперсии
    print("=" * 60)
    print("СРАВНЕНИЕ РЕАЛИЗАЦИЙ PCA")
    print("=" * 60)

    print("\n1. Объясненная дисперсия:")
    print(f"{'Компонента':<10} {'Наша реализация':<20} {'Sklearn':<20} {'Разница':<10}")
    print("-" * 60)

    n_comp = n_components or min(len(pca_custom.explained_variance_ratio_),
                                 len(pca_sklearn.explained_variance_ratio_))

    for i in range(n_comp):
        var_custom = pca_custom.explained_variance_ratio_[i]
        var_sklearn = pca_sklearn.explained_variance_ratio_[i]
        diff = abs(var_custom - var_sklearn)
        print(f"{i + 1:<10} {var_custom:<20.10f} {var_sklearn:<20.10f} {diff:<10.2e}")

    print("\n2. Суммарная объясненная дисперсия:")
    print(f"Наша реализация: {np.sum(pca_custom.explained_variance_ratio_):.6f}")
    print(f"Sklearn: {np.sum(pca_sklearn.explained_variance_ratio_):.6f}")

    # Сравнение преобразованных данных (с точностью до знака)
    print("\n3. Сравнение преобразованных данных:")

    # Проверяем совпадение с точностью до знака (знак собственных векторов может отличаться)
    max_abs_correlation = []
    for i in range(n_comp):
        # Ищем максимальную абсолютную корреляцию между колонками
        corr_matrix = np.corrcoef(X_transformed_custom[:, i],
                                  X_transformed_sklearn[:, i], rowvar=False)
        max_corr = np.abs(corr_matrix[0, 1])
        max_abs_correlation.append(max_corr)
        print(f"Компонента {i + 1}: корреляция = {max_corr:.6f}")

    avg_correlation = np.mean(max_abs_correlation)
    print(f"\nСредняя абсолютная корреляция: {avg_correlation:.6f}")

    # Визуализация сравнения

    return pca_custom, pca_sklearn
