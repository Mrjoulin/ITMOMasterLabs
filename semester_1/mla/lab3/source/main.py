import argparse
import numpy as np
from utils import compare_with_sklearn
from data import run_data_pipeline, load_data_from_csv, train_test_split


def main():
    parser = argparse.ArgumentParser(description='Main script to run data and training pipelines.')
    parser.add_argument(
        '--mode', type=str, default='full', choices=['full', 'data', 'train'],
        help='Mode to run: "full" (data and train), "data" (only data pipeline), or "train" (only training pipeline)'
    )
    parser.add_argument('--with-plotting', action='store_true', help='Enable plotting')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size for data splitting')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--save-path', type=str, default=None, help='Path to save the processed data')
    parser.add_argument('--C', type=float, default=1., help='C value for SVM, default 1')
    parser.add_argument(
        '--data-path', type=str, default='processed_data.csv',
        help='Path where processed data saved (for "train" mode of execution)'
    )

    args = parser.parse_args()

    if args.mode in ['full', 'data']:
        X_train, X_test, y_train, y_test = run_data_pipeline(
            return_split=True,
            test_size=args.test_size,
            random_seed=args.random_seed
        )

        if args.mode == 'data':
            exit()
    else:
        print("Loading preprocessed data...")
        df = load_data_from_csv(args.data_path)
        X_train, X_test, y_train, y_test = train_test_split(
            df, test_size=args.test_size, random_seed=args.random_seed
        )

    print("Train size:", X_train.shape, "Test size:", X_test.shape)

    models, predictions = compare_with_sklearn(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, C=args.C
    )

    if args.with_plotting:
        from utils.plotting import plot_confusion_matrix, visualize_predictions
        plot_confusion_matrix(y_test, predictions)
        visualize_predictions(X_test, y_test, predictions)

    print(f"\nSupport Vectors Info\n")

    for model in models["Our SVM"].values():
        # Анализ опорных векторов для кастомной модели
        if hasattr(model, '_lambdas'):
            print(f"Our SVM with kernel {model.kernel}:")
            print(f"- Количество опорных векторов: {len(model._lambdas)}")
            print(f"- Максимальное lambda: {np.max(model._lambdas):.6f}")
            print(f"- Минимальное lambda: {np.min(model._lambdas):.6f}")


if __name__ == "__main__":
    main()
