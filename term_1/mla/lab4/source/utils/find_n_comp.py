import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from models import CustomPCA


def determine_effective_dimension(X, threshold=0.95):
    """
    Определение эффективной размерности по критерию объясненной дисперсии

    Параметры:
    X: данные
    threshold: порог объясненной дисперсии (по умолчанию 95%)

    Возвращает:
    effective_dim: эффективная размерность
    """
    # Используем нашу реализацию PCA
    pca_custom = CustomPCA()
    pca_custom.fit(X)

    # Кумулятивная объясненная дисперсия
    cumulative_variance = pca_custom.get_cumulative_variance()

    # Находим первую компоненту, где дисперсия превышает порог
    effective_dim = np.argmax(cumulative_variance >= threshold) + 1

    print(f"Общая дисперсия: {np.sum(pca_custom.explained_variance_ratio_):.4f}")
    print(f"Эффективная размерность (при {threshold * 100:.0f}% дисперсии): {effective_dim}")

    # Таблица с информацией о компонентах
    components_info = pd.DataFrame({
        'Компонента': range(1, len(pca_custom.explained_variance_ratio_) + 1),
        'Объясненная дисперсия': pca_custom.explained_variance_ratio_,
        'Кумулятивная дисперсия': cumulative_variance
    })
    print("\nИнформация о главных компонентах:")
    print(components_info.head(40).to_string(index=False))

    return effective_dim
