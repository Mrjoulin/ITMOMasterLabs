import argparse
from utils import compare_with_sklearn, determine_effective_dimension
from data import run_data_pipeline, load_data_from_csv


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
    parser.add_argument('--threshold', type=float, default=0.95, help='Needed explained variance')
    parser.add_argument(
        '--data-path', type=str, default='processed_data.csv',
        help='Path where processed data saved (for "train" mode of execution)'
    )

    args = parser.parse_args()

    if args.mode in ['full', 'data']:
        df = run_data_pipeline(
            return_split=False,
            test_size=args.test_size,
            random_seed=args.random_seed
        )

        if args.mode == 'data':
            exit()
    else:
        print("Loading preprocessed data...")
        df = load_data_from_csv(args.data_path)

    X, y = df.drop("target", axis=1).to_numpy(), df["target"].to_numpy()

    n_components = determine_effective_dimension(X, threshold=args.threshold)

    models = compare_with_sklearn(X=X, n_components=n_components)

    if args.with_plotting:
        from utils.plotting import plot_variance, visualize_pca_space
        plot_variance(*models)
        visualize_pca_space(X, y)


if __name__ == "__main__":
    main()
