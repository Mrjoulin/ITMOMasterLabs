import argparse

from data import run_data_pipeline, load_data_from_csv, train_test_split
from train import train_pipeline, train_sklearn_models
from metrics import evaluate_model


def main():
    parser = argparse.ArgumentParser(description='Main script to run data and training pipelines.')
    parser.add_argument(
        '--mode', type=str, default='full', choices=['full', 'data', 'train'],
        help='Mode to run: "full" (data and train), "data" (only data pipeline), or "train" (only training pipeline)'
    )
    parser.add_argument('--with-plotting', action='store_true', help='Enable plotting')
    parser.add_argument(
        '--loss', type=str, default="log_loss", choices=['binary', 'log_loss', 'sigmoid'],
        help='Loss function to use: "binary", "log_loss", or "sigmoid"'
    )
    parser.add_argument('--num-starts', type=int, default=10, help='Number of starts for multi-start training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--test-size', type=float, default=0.3, help='Test set size for data splitting')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--save-path', type=str, default=None, help='Path to save the processed data')
    parser.add_argument(
        '--data-path', type=str, default='processed_data.csv',
        help='Path where processed data saved (for "train" mode of execution)'
    )

    args = parser.parse_args()

    if args.mode in ['full', 'data']:
        X_train, X_test, y_train, y_test = run_data_pipeline(
            save_path=args.save_path,
            with_plotting=args.with_plotting,
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

    best_model, best_hist = train_pipeline(
        X_train, y_train, X_test, y_test,
        loss=args.loss,
        num_starts=args.num_starts,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    print("\nBest custom model evaluation of functionality (Q):")
    print(f"-- Train Q: {best_hist[0][-1]:.4f}")
    print(f"-- Test Q: {best_hist[1][-1]:.4f}")

    print("\nEvaluating best custom model:")
    conf_matrix = evaluate_model(best_model, X_test, y_test, log_prefix="-- ")
    print("\nConfusion Matrix:")
    print(conf_matrix)

    if args.with_plotting:
        from metrics.plot_metrics import plot_all_hist, plot_margins
        print("\nPlotting metrics...")
        plot_all_hist(*best_hist)
        plot_margins(best_model, X_test, y_test)

    print("\nTraining and evaluating sklearn models for comparison...")
    sklearn_model = train_sklearn_models(X_train, y_train, X_test, y_test, epochs=args.epochs)
    print("\nEvaluating best sklearn model:")
    conf_matrix_sklearn = evaluate_model(sklearn_model, X_test, y_test, log_prefix="-- ")
    print("\nConfusion Matrix (sklearn):")
    print(conf_matrix_sklearn)


if __name__ == '__main__':
    main()
