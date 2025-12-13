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

  # Fetch match data from HLTV URL
  python -m src.main fetch --url https://www.hltv.org/matches/2388125/spirit-vs-falcons-...

  # Predict match outcome from fetched data
  python -m src.main predict --fetched-data data/fetched/match_2388125.json

  # Predict match outcome manually (uses preprocessed data)
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

    # Fetch command
    fetch_parser = subparsers.add_parser(
        "fetch", help="Fetch match data from HLTV URL"
    )
    fetch_parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="HLTV match URL (e.g., https://www.hltv.org/matches/2388125/...)",
    )
    fetch_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path (default: data/fetched/match_{id}.json)",
    )
    fetch_parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Root directory of the project",
    )
    fetch_parser.add_argument(
        "--quiet", action="store_true", help="Suppress verbose output"
    )

    # Predict command
    predict_parser = subparsers.add_parser(
        "predict", help="Predict match outcome from fetched data or manual parameters"
    )
    predict_parser.add_argument(
        "--fetched-data",
        type=Path,
        default=None,
        help="Path to JSON file with fetched match data (from fetch command)",
    )
    predict_parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="HLTV match URL (deprecated - use fetch command first)",
    )
    predict_parser.add_argument(
        "--team-a",
        type=str,
        default=None,
        help="Name of team A (required if --fetched-data not provided)",
    )
    predict_parser.add_argument(
        "--team-b",
        type=str,
        default=None,
        help="Name of team B (required if --fetched-data not provided)",
    )
    predict_parser.add_argument(
        "--map",
        type=str,
        default=None,
        help="Map name (required if --fetched-data not provided)",
    )
    predict_parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Match date in YYYY-MM-DD format (required if --fetched-data not provided)",
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
    predict_parser.add_argument(
        "--bet", action="store_true", help="Calculate betting recommendations (will prompt for odds)"
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

        model, best_params, eval_results, calibration_data = train_xgboost_model(
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
        
        # Save model and calibration data
        import pickle
        project_root = getattr(args, 'project_root', None) or Path(__file__).resolve().parent.parent
        models_dir = project_root / "models"
        models_dir.mkdir(exist_ok=True)
        model_path = models_dir / "xgboost_model.pkl"
        calibration_path = models_dir / "calibration_data.pkl"
        
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        
        if calibration_data:
            with open(calibration_path, "wb") as f:
                pickle.dump(calibration_data, f)
            if not args.quiet:
                logger.info(f"\nModel saved to: {model_path}")
                logger.info(f"Calibration data saved to: {calibration_path}")
        else:
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

            model, best_params, eval_results, calibration_data = train_xgboost_model(
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

    elif args.command == "fetch":
        from .inference.fetch_data import fetch_match_data
        
        project_root = args.project_root or Path(__file__).resolve().parent.parent
        
        fetched_data = fetch_match_data(
            match_url=args.url,
            output_file=args.output,
            project_root=project_root,
            verbose=not args.quiet
        )
        
        if not fetched_data:
            logger.error("Failed to fetch match data")
            sys.exit(1)
        
        if not args.quiet:
            logger.info("\nFetch completed successfully!")
            if fetched_data and '_file_path' in fetched_data:
                file_path = fetched_data['_file_path']
                logger.info(f"Fetched data saved to: {file_path}")
                # Output the full command to run
                python_cmd = sys.executable or "python"
                full_command = f"{python_cmd} -m src.main predict --fetched-data {file_path}"
                logger.info(f"\nRun this command to make predictions:")
                logger.info(f"  {full_command}")
            else:
                logger.info(f"Use the predict command with --fetched-data to make predictions")

    elif args.command == "predict":
        from .inference import predict_match
        from .inference.betting import american_to_decimal, calculate_betting_recommendation
        from .eval.calibration import get_calibration_accuracy
        
        project_root = args.project_root or Path(__file__).resolve().parent.parent
        model_path = args.model_path or (project_root / "models" / "xgboost_model.pkl")
        
        result = predict_match(
            fetched_data_file=args.fetched_data,
            match_url=args.url,
            team_a=args.team_a,
            team_b=args.team_b,
            map_name=args.map,
            match_date=args.date,
            model_path=model_path if model_path.exists() else None,
            project_root=project_root,
            verbose=not args.quiet
        )
        
        # Handle betting calculations if --bet flag is set
        betting_info = None
        if args.bet and result:
            # Load calibration data
            calibration_data = None
            calibration_file = project_root / "models" / "calibration_data.pkl"
            if calibration_file.exists():
                try:
                    import pickle
                    with open(calibration_file, 'rb') as f:
                        calibration_data = pickle.load(f)
                except Exception:
                    pass
            
            # Prompt for odds
            print("\n" + "=" * 80)
            print("Betting Odds Input")
            print("=" * 80)
            
            team_a_name = result.get('team_a', 'Team A')
            team_b_name = result.get('team_b', 'Team B')
            
            # Get odds for team A
            while True:
                odds_a_input = input(f"Enter odds for {team_a_name} (American format, e.g., +100 or -120): ").strip()
                decimal_odds_a = american_to_decimal(odds_a_input)
                if decimal_odds_a:
                    break
                print("Invalid format. Please enter odds in American format (e.g., +100, -120, +150)")
            
            # Get odds for team B
            while True:
                odds_b_input = input(f"Enter odds for {team_b_name} (American format, e.g., +100 or -120): ").strip()
                decimal_odds_b = american_to_decimal(odds_b_input)
                if decimal_odds_b:
                    break
                print("Invalid format. Please enter odds in American format (e.g., +100, -120, +150)")
            
            # Calculate betting recommendations
            if 'predictions_by_map' in result:
                # Multiple maps - calculate for each map
                betting_info = {}
                for map_name, pred in result['predictions_by_map'].items():
                    team_a_prob = pred['team_a_win_probability']
                    team_b_prob = pred['team_b_win_probability']
                    
                    team_a_calib = None
                    team_b_calib = None
                    if calibration_data:
                        team_a_calib = get_calibration_accuracy(calibration_data, team_a_prob)
                        team_b_calib = get_calibration_accuracy(calibration_data, team_b_prob)
                    
                    betting_info[map_name] = calculate_betting_recommendation(
                        team_a_prob=team_a_prob,
                        team_b_prob=team_b_prob,
                        team_a_odds=decimal_odds_a,
                        team_b_odds=decimal_odds_b,
                        team_a_calib_acc=team_a_calib,
                        team_b_calib_acc=team_b_calib
                    )
            else:
                # Single map
                team_a_prob = result['team_a_win_probability']
                team_b_prob = result['team_b_win_probability']
                
                team_a_calib = None
                team_b_calib = None
                if calibration_data:
                    team_a_calib = get_calibration_accuracy(calibration_data, team_a_prob)
                    team_b_calib = get_calibration_accuracy(calibration_data, team_b_prob)
                
                betting_info = calculate_betting_recommendation(
                    team_a_prob=team_a_prob,
                    team_b_prob=team_b_prob,
                    team_a_odds=decimal_odds_a,
                    team_b_odds=decimal_odds_b,
                    team_a_calib_acc=team_a_calib,
                    team_b_calib_acc=team_b_calib
                )
        
        if result:
            # Check if we have predictions for multiple maps
            if 'predictions_by_map' in result:
                # Sort by team_a win probability (highest first)
                sorted_maps = sorted(
                    result['predictions_by_map'].items(),
                    key=lambda x: x[1]['team_a_win_probability'],
                    reverse=True
                )
                
                print(f"\n{result['team_a']} vs {result['team_b']} - {result['match_date']}")
                print("=" * 80)
                for map_name, pred in sorted_maps:
                    calib_str = f" (Model Accuracy: {pred.get('calibration_accuracy', 0):.1%})" if pred.get('calibration_accuracy') else ""
                    betting_str = ""
                    if betting_info and map_name in betting_info:
                        bet_info = betting_info[map_name]
                        if bet_info.get('best_bet'):
                            best_team = bet_info['best_bet']
                            profit = bet_info[best_team]['expected_profit']
                            if profit > 0:
                                betting_str = f" | Expected Profit: {profit:+.1%}"
                    print(f"{map_name:12} | {result['team_a']:20} {pred['team_a_win_probability']:6.2%} | {result['team_b']:20} {pred['team_b_win_probability']:6.2%} | Winner: {pred['predicted_winner']}{calib_str}{betting_str}")
                print("=" * 80)
                
                # Display betting recommendations if --bet was used
                if betting_info:
                    print("\nBetting Recommendations:")
                    print("=" * 80)
                    for map_name, pred in sorted_maps:
                        if map_name in betting_info:
                            bet_info = betting_info[map_name]
                            team_a_bet = bet_info.get('team_a', {})
                            team_b_bet = bet_info.get('team_b', {})
                            
                            print(f"\n{map_name}:")
                            if team_a_bet.get('odds'):
                                profit_a = team_a_bet.get('expected_profit', 0)
                                rec_a = team_a_bet.get('recommendation', 'avoid')
                                odds_a = team_a_bet.get('odds', 0)
                                print(f"  {result['team_a']:30} | Odds: {odds_a:.2f} | Expected Profit: {profit_a:+.2%} | {rec_a.upper()}")
                            
                            if team_b_bet.get('odds'):
                                profit_b = team_b_bet.get('expected_profit', 0)
                                rec_b = team_b_bet.get('recommendation', 'avoid')
                                odds_b = team_b_bet.get('odds', 0)
                                print(f"  {result['team_b']:30} | Odds: {odds_b:.2f} | Expected Profit: {profit_b:+.2%} | {rec_b.upper()}")
                            
                            best_bet = bet_info.get('best_bet')
                            if best_bet:
                                best_team_name = result[best_bet]
                                best_profit = bet_info[best_bet]['expected_profit']
                                print(f"  Best Bet: {best_team_name} (Expected Profit: {best_profit:+.2%})")
                            else:
                                print(f"  No profitable bets found")
                    print("=" * 80)
            else:
                calib_str = f"\nModel Accuracy: {result.get('calibration_accuracy', 0):.1%}" if result.get('calibration_accuracy') else ""
                betting_str = ""
                if betting_info:
                    bet_info = betting_info
                    if bet_info.get('best_bet'):
                        best_team = bet_info['best_bet']
                        profit = bet_info[best_team]['expected_profit']
                        if profit > 0:
                            betting_str = f" | Expected Profit: {profit:+.1%}"
                
                print(f"\n{result['team_a']} vs {result['team_b']} - {result['map']} ({result['match_date']})")
                print("=" * 80)
                print(f"Predicted Winner: {result['predicted_winner']}{betting_str}")
                print(f"{result['team_a']:30} {result['team_a_win_probability']:6.2%}")
                print(f"{result['team_b']:30} {result['team_b_win_probability']:6.2%}")
                if calib_str:
                    print(calib_str)
                print("=" * 80)
                
                # Display betting recommendations if --bet was used
                if betting_info:
                    print("\nBetting Recommendations:")
                    print("=" * 80)
                    team_a_bet = betting_info.get('team_a', {})
                    team_b_bet = betting_info.get('team_b', {})
                    
                    if team_a_bet.get('odds'):
                        profit_a = team_a_bet.get('expected_profit', 0)
                        rec_a = team_a_bet.get('recommendation', 'avoid')
                        odds_a = team_a_bet.get('odds', 0)
                        print(f"{result['team_a']:30} | Odds: {odds_a:.2f} | Expected Profit: {profit_a:+.2%} | {rec_a.upper()}")
                    
                    if team_b_bet.get('odds'):
                        profit_b = team_b_bet.get('expected_profit', 0)
                        rec_b = team_b_bet.get('recommendation', 'avoid')
                        odds_b = team_b_bet.get('odds', 0)
                        print(f"{result['team_b']:30} | Odds: {odds_b:.2f} | Expected Profit: {profit_b:+.2%} | {rec_b.upper()}")
                    
                    best_bet = betting_info.get('best_bet')
                    if best_bet:
                        best_team_name = result[best_bet]
                        best_profit = betting_info[best_bet]['expected_profit']
                        print(f"\nBest Bet: {best_team_name} (Expected Profit: {best_profit:+.2%})")
                    else:
                        print(f"\nNo profitable bets found")
                    print("=" * 80)
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
