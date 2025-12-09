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

  # Predict match outcome from HLTV URL
  python -m src.main predict --url https://www.hltv.org/matches/2388125/spirit-vs-falcons-...

  # Predict match outcome manually
  python -m src.main predict --team-a "Team Spirit" --team-b "Team Falcons" --map "Mirage" --date "2025-01-15"
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

    # Predict command
    predict_parser = subparsers.add_parser(
        "predict", help="Predict match outcome from HLTV URL or match details"
    )
    predict_parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="HLTV match URL (e.g., https://www.hltv.org/matches/2388125/...)",
    )
    predict_parser.add_argument(
        "--team-a",
        type=str,
        default=None,
        help="Name of team A (required if --url not provided)",
    )
    predict_parser.add_argument(
        "--team-b",
        type=str,
        default=None,
        help="Name of team B (required if --url not provided)",
    )
    predict_parser.add_argument(
        "--map",
        type=str,
        default=None,
        help="Map name (required if --url not provided)",
    )
    predict_parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Match date in YYYY-MM-DD format (required if --url not provided)",
    )
    predict_parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to saved model (default: models/xgboost_model.pkl)",
    )
    predict_parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Root directory of the project",
    )
    predict_parser.add_argument(
        "--quiet", action="store_true", help="Suppress verbose output"
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
        X_train, X_test, y_train, y_test, original_distribution = load_and_prepare_data(
            data_file=args.data,
            balance_classes=not args.no_balance,
            random_seed=args.seed,
            verbose=not args.quiet,
        )

        model, best_params, eval_results = train_xgboost_model(
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
        
        # Save model
        import pickle
        project_root = getattr(args, 'project_root', None) or Path(__file__).resolve().parent.parent
        models_dir = project_root / "models"
        models_dir.mkdir(exist_ok=True)
        model_path = models_dir / "xgboost_model.pkl"
        
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        
        if not args.quiet:
            logger.info(f"\nModel saved to: {model_path}")

    elif args.command == "visualize":
        import pickle

        if args.no_train:
            if args.model_path is None or not args.model_path.exists():
                logger.error("--model-path is required when using --no-train")
                sys.exit(1)

            logger.info("Loading model from file...")
            with open(args.model_path, "rb") as f:
                model = pickle.load(f)

            X_train, X_test, y_train, y_test, original_distribution = load_and_prepare_data(
                data_file=args.data,
                balance_classes=True,
                random_seed=args.seed,
                verbose=not args.quiet,
            )
            eval_results = None
        else:
            logger.info("Training model for visualization...")
            X_train, X_test, y_train, y_test, original_distribution = load_and_prepare_data(
                data_file=args.data,
                balance_classes=True,
                random_seed=args.seed,
                verbose=not args.quiet,
            )

            model, best_params, eval_results = train_xgboost_model(
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
            eval_results=eval_results,
            X_train=X_train if not args.no_train else None,
            y_train=y_train if not args.no_train else None,
            original_distribution=original_distribution,
        )

    elif args.command == "predict":
        from .inference import predict_match
        
        project_root = args.project_root or Path(__file__).resolve().parent.parent
        model_path = args.model_path or (project_root / "models" / "xgboost_model.pkl")
        
        result = predict_match(
            match_url=args.url,
            team_a=args.team_a,
            team_b=args.team_b,
            map_name=args.map,
            match_date=args.date,
            model_path=model_path if model_path.exists() else None,
            project_root=project_root,
            verbose=not args.quiet
        )
        
        if result:
            logger.info("\n" + "="*60)
            logger.info("MATCH PREDICTION RESULTS")
            logger.info("="*60)
            logger.info(f"Match: {result['team_a']} vs {result['team_b']}")
            logger.info(f"Map: {result['map'] or 'All Maps (map not decided)'}")
            logger.info(f"Date: {result['match_date']}")
            logger.info("")
            
            # Check if we have predictions for multiple maps
            if 'predictions_by_map' in result:
                logger.info("Predictions for all maps:")
                logger.info("")
                # Sort by team_a win probability (highest first)
                sorted_maps = sorted(
                    result['predictions_by_map'].items(),
                    key=lambda x: x[1]['team_a_win_probability'],
                    reverse=True
                )
                for map_name, pred in sorted_maps:
                    logger.info(f"{map_name}:")
                    logger.info(f"  {result['team_a']}: {pred['team_a_win_probability']:.2%}")
                    logger.info(f"  {result['team_b']}: {pred['team_b_win_probability']:.2%}")
                    logger.info(f"  Predicted Winner: {pred['predicted_winner']}")
                    logger.info("")
            else:
                logger.info(f"Predicted Winner: {result['predicted_winner']}")
                logger.info("")
                logger.info("Win Probabilities:")
                logger.info(f"  {result['team_a']}: {result['team_a_win_probability']:.2%}")
                logger.info(f"  {result['team_b']}: {result['team_b_win_probability']:.2%}")
            
            logger.info("="*60)
        else:
            logger.error("Failed to generate prediction")
            sys.exit(1)

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
