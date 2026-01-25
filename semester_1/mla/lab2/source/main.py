import argparse
from selection.prototype_selector import PrototypeSelector
from utils import find_best_k_loo, compare_with_sklearn
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
    parser.add_argument('--k-max', type=int, default=20, help='Max k value for KNN in LLO search')
    parser.add_argument(
        '--k-prototype', type=int, default=3,
        help='k value for prototype selection process (max m in compactness profile)'
    )
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

    # Find best k by LOO
    best_k = find_best_k_loo(X_train, y_train, k_end=args.k_max, plot_graph=args.with_plotting)
    compare_with_sklearn(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, best_k=best_k
    )

    prototype_selector = PrototypeSelector(k=args.k_prototype)
    prototype_selector.fit(X_train, y_train)
    X_prototypes, y_prototypes = prototype_selector.get_prototypes()

    print(f"\nSelection results:")
    print(f"Original size: {X_train.shape[0]} samples")
    print(f"After selection: {X_prototypes.shape[0]} samples")
    print(f"Compression ratio: {X_train.shape[0] / X_prototypes.shape[0]:.2f}x")

    if args.with_plotting:
        from utils.plotting import (
            plot_compactness_full, plot_prototype_selection_process, visualize_prototype_selection
        )
        plot_compactness_full(X_train, y_train)
        plot_prototype_selection_process(prototype_selector.history)
        visualize_prototype_selection(X_train, y_train, X_prototypes, y_prototypes)

    # Compare again with prototypes
    compare_with_sklearn(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, best_k=best_k,
        X_prototypes=X_prototypes, y_prototypes=y_prototypes
    )


if __name__ == "__main__":
    main()
