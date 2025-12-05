"""Main entry point for the ML project."""

import argparse
import sys
import logging
from pathlib import Path

from .data.preprocess import run_preprocessing_pipeline
from .data.final_features import create_final_features
from .run.train import load_and_prepare_data, train_xgboost_model
from .eval.metrics import evaluate_model
from .eval.visualize import create_all_visualizations


def setup_logging(verbose: bool = True):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main(argv=None):
    """Main CLI entry point."""
    argv = argv or sys.argv[1:]
    parser = argparse.ArgumentParser(
        description="CS2 Match Prediction ML Project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full preprocessing pipeline
  python -m src.main preprocess

  # Train model (includes evaluation metrics)
  python -m src.main train --data data/preprocessed/final_features.csv

  # Create visualizations (confusion matrix, ROC curve, feature importance)
  python -m src.main visualize --data data/preprocessed/final_features.csv
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Preprocess command
    preprocess_parser = subparsers.add_parser(
        "preprocess", help="Run preprocessing pipeline"
    )
    preprocess_parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root directory (default: auto-detect)",
    )
    preprocess_parser.add_argument(
        "--quiet", action="store_true", help="Suppress verbose output"
    )

    # Train command
    train_parser = subparsers.add_parser("train", help="Train XGBoost model")
    train_parser.add_argument(
        "--data", type=Path, required=True, help="Path to final_features.csv"
    )
    train_parser.add_argument(
        "--no-balance", action="store_true", help="Don't balance classes"
    )
    train_parser.add_argument(
        "--no-tuning", action="store_true", help="Skip hyperparameter tuning"
    )
    train_parser.add_argument(
        "--n-iter",
        type=int,
        default=50,
        help="Number of hyperparameter combinations to try (default: 50)",
    )
    train_parser.add_argument(
        "--seed", type=int, default=49, help="Random seed (default: 49)"
    )
    train_parser.add_argument(
        "--quiet", action="store_true", help="Suppress verbose output"
    )

    # Visualize command
    visualize_parser = subparsers.add_parser(
        "visualize", help="Create visualizations for model evaluation"
    )
    visualize_parser.add_argument(
        "--data", type=Path, required=True, help="Path to final_features.csv"
    )
    visualize_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save plots (default: plots/)",
    )
    visualize_parser.add_argument(
        "--no-train",
        action="store_true",
        help="Skip training, use existing model (requires --model-path)",
    )
    visualize_parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to saved model (required if --no-train)",
    )
    visualize_parser.add_argument(
        "--seed", type=int, default=49, help="Random seed (default: 49)"
    )
    visualize_parser.add_argument(
        "--quiet", action="store_true", help="Suppress verbose output"
    )
    visualize_parser.add_argument(
        "--show", action="store_true", help="Display plots interactively"
    )

    # About command
    subparsers.add_parser("about", help="Show project info")

    args = parser.parse_args(argv)

    # Setup logging (only if command has quiet attribute)
    has_quiet = hasattr(args, "quiet")
    setup_logging(verbose=not has_quiet or not args.quiet)
    logger = logging.getLogger(__name__)

    if args.command == "preprocess":
        run_preprocessing_pipeline(
            project_root=args.project_root, verbose=not args.quiet
        )

        # Also create final_features.csv
        project_root = args.project_root or Path(__file__).resolve().parent.parent
        match_features_file = (
            project_root / "data" / "preprocessed" / "match_features.csv"
        )
        final_features_file = (
            project_root / "data" / "preprocessed" / "final_features.csv"
        )

        if match_features_file.exists():
            create_final_features(
                match_features_file=match_features_file,
                output_file=final_features_file,
                verbose=not args.quiet,
            )

    elif args.command == "train":
        X_train, X_test, y_train, y_test = load_and_prepare_data(
            data_file=args.data,
            balance_classes=not args.no_balance,
            random_seed=args.seed,
            verbose=not args.quiet,
        )

        model, best_params = train_xgboost_model(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            hyperparameter_tuning=not args.no_tuning,
            n_iter=args.n_iter,
            random_seed=args.seed,
            verbose=not args.quiet,
        )

        # Evaluate on test set
        y_pred = model.predict(X_test)
        evaluate_model(y_test.values, y_pred, verbose=not args.quiet)

    elif args.command == "visualize":
        import pickle

        if args.no_train:
            if args.model_path is None or not args.model_path.exists():
                logger.error("--model-path is required when using --no-train")
                sys.exit(1)

            logger.info("Loading model from file...")
            with open(args.model_path, "rb") as f:
                model = pickle.load(f)

            X_train, X_test, y_train, y_test = load_and_prepare_data(
                data_file=args.data,
                balance_classes=True,
                random_seed=args.seed,
                verbose=not args.quiet,
            )
        else:
            logger.info("Training model for visualization...")
            X_train, X_test, y_train, y_test = load_and_prepare_data(
                data_file=args.data,
                balance_classes=True,
                random_seed=args.seed,
                verbose=not args.quiet,
            )

            model, best_params = train_xgboost_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                hyperparameter_tuning=False,  # Skip tuning for faster visualization
                random_seed=args.seed,
                verbose=not args.quiet,
            )

        # Get predictions and probabilities
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class

        # Evaluate model
        logger.info("")
        evaluate_model(y_test.values, y_pred, verbose=not args.quiet)

        # Create visualizations
        output_dir = args.output_dir or Path("plots")
        create_all_visualizations(
            model=model,
            X_test=X_test,
            y_test=y_test.values,
            y_pred=y_pred,
            y_proba=y_proba,
            output_dir=output_dir,
            show=args.show,
        )

    elif args.command == "about":
        print("CS2 Match Prediction ML Project")
        print("A machine learning project for predicting CS2 match outcomes")
        print("using team statistics, player performance, and historical data.")
        print("\nProject structure:")
        print("  src/data/     - Data preprocessing modules")
        print("  src/run/      - Model training scripts")
        print("  src/eval/     - Evaluation and metrics")
        print("  notebooks/    - Jupyter notebooks for exploration")
        print("  data/         - Raw and processed data")
        print("\nSee README.md for more details.")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
