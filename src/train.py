"""
End-to-end AML training pipeline.

Stages (run in order by default):
  1. build   – join raw data into a single feature table
  2. split   – stratified 70/15/15 train/valid/test split
  3. train   – fit the selected model
  4. eval    – evaluate on validation and test sets

Models:
  baseline   – logistic regression (default)
  xgb        – XGBoost + SMOTE

Usage:
  python -m src.train                            # logistic regression, all stages
  python -m src.train --model xgb               # XGBoost + SMOTE, all stages
  python -m src.train --model xgb --skip-build  # skip data build
  python -m src.train --stages train eval        # run specific stages only
"""

import argparse
import time

from src.data import build_table, make_splits
from src.model import train_baseline, train_xgb, evaluate


ALL_STAGES = ["build", "split", "train", "eval"]
MODELS = {
    "baseline": (train_baseline.main, "logreg_baseline"),
    "xgb":      (train_xgb.main,      "xgb_smote"),
}


def run_stage(name: str, fn, *args, **kwargs):
    print(f"\n{'='*60}")
    print(f"  STAGE: {name.upper()}")
    print(f"{'='*60}")
    t0 = time.time()
    fn(*args, **kwargs)
    elapsed = time.time() - t0
    print(f"  [done in {elapsed:.1f}s]")


def main():
    parser = argparse.ArgumentParser(description="AML end-to-end training pipeline")
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=ALL_STAGES,
        default=ALL_STAGES,
        metavar="STAGE",
        help=f"Stages to run (default: all). Choices: {ALL_STAGES}",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip the 'build' stage (feature table already exists)",
    )
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        default="baseline",
        help="Model to train: 'baseline' (logistic regression) or 'xgb' (XGBoost + SMOTE). Default: baseline",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Override the saved artifact name (default: derived from --model)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splits and training (default: 42)",
    )
    args = parser.parse_args()

    train_fn, default_model_name = MODELS[args.model]
    model_name = args.model_name or default_model_name

    stages = args.stages
    if args.skip_build and "build" in stages:
        stages = [s for s in stages if s != "build"]

    t_total = time.time()

    if "build" in stages:
        run_stage("build", build_table.main)

    if "split" in stages:
        run_stage("split", make_splits.main, seed=args.seed)

    if "train" in stages:
        run_stage(f"train ({args.model})", train_fn, model_name=model_name, seed=args.seed)

    if "eval" in stages:
        run_stage("eval (valid)", evaluate.main, model_name=model_name, split="valid")
        run_stage("eval (test)", evaluate.main, model_name=model_name, split="test")

    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE  ({time.time() - t_total:.1f}s total)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
