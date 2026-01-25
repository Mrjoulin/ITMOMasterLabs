import argparse
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

from data import run_data_pipeline, load_data_from_csv, train_test_split
from models import LogisticRegression
from utils import train_eval_model, compare_with_sklearn


def main():
    parser = argparse.ArgumentParser(description='Main script to run data and training pipelines.')
    parser.add_argument(
        '--mode', type=str, default='full', choices=['full', 'data', 'train'],
        help='Mode to run: "full" (data and train), "data" (only data pipeline), or "train" (only training pipeline)'
    )
    parser.add_argument('--with-plotting', action='store_true', help='Enable plotting')
    parser.add_argument('--test-size', type=float, default=0.3, help='Test set size for data splitting')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--save-path', type=str, default=None, help='Path to save the processed data')
    parser.add_argument('--max-iter', type=int, default=200, help='Max iterations for logistic regression')
    parser.add_argument('--tol', type=float, default=0.001, help='Tolerance for logistic regression convergence')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for logistic regression')
    parser.add_argument(
        '--data-path', type=str, default='processed_data.csv',
        help='Path where processed data saved (for "train" mode of execution)'
    )

    args = parser.parse_args()

    if args.mode in ['full', 'data']:
        prepared_data = run_data_pipeline(
            return_split=args.mode == 'full',
            test_size=args.test_size,
            random_seed=args.random_seed,
            save_path=args.save_path
        )

        if args.mode == 'data':
            exit()
        X_train, X_test, y_train, y_test = prepared_data
    else:
        print("Loading preprocessed data...")
        df = load_data_from_csv(args.data_path)
        X_train, X_test, y_train, y_test = train_test_split(
            df, test_size=args.test_size, random_seed=args.random_seed
        )

    models = {
        "Custom NR": LogisticRegression(solver='nr', max_iter=args.max_iter, tol=args.tol, learning_rate=args.lr),
        "Custom IRLS": LogisticRegression(solver='irls', max_iter=args.max_iter, tol=args.tol, learning_rate=args.lr),
        "Sklearn": SklearnLogisticRegression(max_iter=args.max_iter, tol=args.tol)
    }

    print("\nTraining our Logistic Regression (Newton-Raphson)...")
    cm_nr = train_eval_model(models["Custom NR"], X_train, y_train, X_test, y_test)
    print("\nTraining our Logistic Regression (IRLS)...")
    cm_irls = train_eval_model(models["Custom IRLS"], X_train, y_train, X_test, y_test)
    print("\nTraining Sklearn's Logistic Regression...")
    cm_sklearn = train_eval_model(models["Sklearn"], X_train, y_train, X_test, y_test)

    models_scores = compare_with_sklearn(models, X_test, y_test)

    if args.with_plotting:
        from utils.plotting import plot_confusion_matrix, plot_roc_curve

        plot_confusion_matrix(cm_nr, title="Confusion Matrix (Custom NR)", img_name="cm_custom_nr.png")
        plot_confusion_matrix(cm_irls, title="Confusion Matrix (Custom IRLS)", img_name="cm_custom_irls.png")
        plot_confusion_matrix(cm_sklearn, title="Confusion Matrix (Sklearn)", img_name="cm_sklearn.png")

        plot_roc_curve(y_test, models_scores, img_name="roc_curve_comparison.png")


if __name__ == "__main__":
    main()
