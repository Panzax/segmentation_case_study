#!/usr/bin/env python3
"""
Run separate learning rate sweeps for each feature size.

This script creates and optionally runs WandB sweeps to find the optimal
learning rate for each feature size independently.

Usage:
    # Initialize sweeps for all feature sizes
    python sweep_lr_per_feature_size.py \
        --images_dir /path/to/images \
        --labels_dir /path/to/labels \
        --output_dir /path/to/output \
        [--feature_sizes 12 24 48] \
        [--learning_rates 1e-4 5e-4 1e-3 2e-3] \
        [--run_agent]

    # Run agent for a specific feature size sweep
    python sweep_lr_per_feature_size.py \
        --sweep_id <sweep_id> \
        --run_agent \
        --images_dir ... --labels_dir ... --output_dir ...
"""

import argparse
import ast
import subprocess
import sys
from pathlib import Path

try:
    import wandb
    import yaml
except ImportError as e:
    print(f"Error: Required packages not installed: {e}")
    print("Install with: pip install wandb pyyaml")
    sys.exit(1)


def create_lr_sweep_config(
    feature_size: int,
    learning_rates: list,
    images_dir: str,
    labels_dir: str,
    output_dir: str,
    val_images_dir: str = None,
    val_labels_dir: str = None,
    device: str = "cuda:0",
    max_epochs: int = 50,
    model_name: str = "SwinUNetR_Mlp_LeakyReLU",
    depths: list = [2, 2, 2, 2],
    batch_size: int = 1,
    loss_function: str = "Generalized Dice",
    scheduler_factor: float = 0.5,
    scheduler_patience: int = 10,
    method: str = "grid",
) -> dict:
    """Create a sweep config focused on learning rate optimization for a specific feature size."""
    script_path = Path(__file__).parent / "train_cellseg3d_swinunetr.py"
    config = {
        "program": str(script_path),
        "method": method,
        "metric": {
            "name": "Validation/Batch Loss",
            "goal": "minimize",
        },
        "parameters": {
            "images_dir": {"value": str(images_dir)},
            "labels_dir": {"value": str(labels_dir)},
            "output_dir": {"value": str(output_dir)},
            "device": {"value": device},
            "max_epochs": {"value": max_epochs},
            "model_name": {"value": model_name},
            "feature_size": {"value": feature_size},  # Fixed for this sweep
            # NOTE: we deliberately do NOT include 'depths' here.
            # Passing it via CLI conflicts with argparse's `nargs=4` handling and
            # causes `expected 4 arguments` errors when wandb serializes the list.
            # Depths will instead use the training script's CLI default or any
            # explicit `--depths` you pass when running without sweeps.
            "batch_size": {"value": batch_size},
            "learning_rate": {"values": learning_rates},  # Sweep this
            "loss_function": {"value": loss_function},
            # Scheduler hyperparameters are fixed in code; we don't expose them
            # as sweep parameters to avoid wandb injecting unknown CLI flags.
        },
    }
    if val_images_dir:
        config["parameters"]["val_images_dir"] = {"value": str(val_images_dir)}
    if val_labels_dir:
        config["parameters"]["val_labels_dir"] = {"value": str(val_labels_dir)}
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Run learning rate sweeps for each feature size"
    )
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--labels_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--val_images_dir", type=str, default=None)
    parser.add_argument("--val_labels_dir", type=str, default=None)
    parser.add_argument(
        "--feature_sizes",
        type=int,
        nargs="+",
        default=[12, 24, 48],
        help="Feature sizes to sweep (default: 12 24 48)",
    )
    parser.add_argument(
        "--learning_rates",
        type=float,
        nargs="+",
        default=[1e-4, 5e-4, 1e-3, 2e-3, 5e-3],
        help="Learning rates to test (default: 1e-4 5e-4 1e-3 2e-3 5e-3)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="SwinUNetR_Mlp_LeakyReLU",
        help="Model name to use (default: SwinUNetR_Mlp_LeakyReLU)",
    )
    parser.add_argument(
        "--depths",
        type=int,
        nargs=4,
        default=[2, 2, 2, 2],
        metavar="DEPTH",
        help="Model depths [d1 d2 d3 d4] (default: 2 2 2 2)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size (default: 1)",
    )
    parser.add_argument(
        "--loss_function",
        type=str,
        default="Generalized Dice",
        help="Loss function (default: Generalized Dice)",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--run_agent", action="store_true")
    parser.add_argument("--sweep_id", type=str, default=None)
    parser.add_argument(
        "--project",
        type=str,
        default="CellSeg3D",
        help="WandB project name",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="grid",
        choices=["grid", "random", "bayes"],
        help="Sweep method (default: grid)",
    )

    args = parser.parse_args()

    script_dir = Path(__file__).parent
    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    output_dir = Path(args.output_dir)

    # Get wandb entity (username/team) - required for project queries
    try:
        viewer = wandb.api.viewer()
        # viewer() returns a dict, extract username/entity
        if isinstance(viewer, dict):
            entity = viewer.get("username") or viewer.get("entity")
        else:
            entity = str(viewer) if viewer else None
    except Exception:
        entity = None

    if not images_dir.exists():
        raise ValueError(f"Images directory does not exist: {images_dir}")
    if not labels_dir.exists():
        raise ValueError(f"Labels directory does not exist: {labels_dir}")

    val_images_dir = Path(args.val_images_dir) if args.val_images_dir else None
    val_labels_dir = Path(args.val_labels_dir) if args.val_labels_dir else None

    if (val_images_dir is not None) ^ (val_labels_dir is not None):
        raise ValueError("Both --val_images_dir and --val_labels_dir must be provided together")

    if args.sweep_id is not None:
        # Run agent for existing sweep
        print(f"Running agent for sweep: {args.sweep_id}")
        print("=" * 60)
        wandb.agent(
            sweep_id=args.sweep_id,
            project=args.project,
            entity=entity,
            count=args.count,
        )
    else:
        # Create sweeps for each feature size
        print("=" * 60)
        print("Creating Learning Rate Sweeps for Each Feature Size")
        print("=" * 60)
        print(f"Project: {args.project}")
        print(f"Feature sizes: {args.feature_sizes}")
        print(f"Learning rates: {args.learning_rates}")
        print(f"Model: {args.model_name}")
        print(f"Depths: {args.depths}")
        print(f"Batch size: {args.batch_size}")
        print(f"Method: {args.method}")
        print("=" * 60)

        sweep_ids = {}
        for feature_size in args.feature_sizes:
            print(f"\nCreating sweep for feature_size={feature_size}...")
            sweep_config = create_lr_sweep_config(
                feature_size=feature_size,
                learning_rates=args.learning_rates,
                images_dir=str(images_dir),
                labels_dir=str(labels_dir),
                output_dir=str(output_dir),
                val_images_dir=str(val_images_dir) if val_images_dir else None,
                val_labels_dir=str(val_labels_dir) if val_labels_dir else None,
                device=args.device,
                max_epochs=args.max_epochs,
                model_name=args.model_name,
                depths=args.depths,
                batch_size=args.batch_size,
                loss_function=args.loss_function,
                method=args.method,
            )

            sweep_id = wandb.sweep(sweep_config, project=args.project, entity=entity)
            sweep_ids[feature_size] = sweep_id

            print(f"  Sweep ID: {sweep_id}")
            print(f"  View at: https://wandb.ai/{entity}/sweeps/{sweep_id}")

            if args.run_agent:
                print(f"  Starting agent for feature_size={feature_size}...")
                wandb.agent(
                    sweep_id=sweep_id,
                    project=args.project,
                    entity=entity,
                    count=args.count,
                )

        print("\n" + "=" * 60)
        print("All sweeps created!")
        print("=" * 60)
        print("\nSweep IDs by feature size:")
        for feature_size, sweep_id in sweep_ids.items():
            print(f"  feature_size={feature_size}: {sweep_id}")

        if not args.run_agent:
            print("\nTo run agents, use:")
            for feature_size, sweep_id in sweep_ids.items():
                print(f"  # For feature_size={feature_size}:")
                print(f"  wandb agent {args.project}/{sweep_id}")
                print(
                    f"  # Or: python sweep_lr_per_feature_size.py --run_agent --sweep_id {sweep_id} "
                    f"--images_dir {images_dir} --labels_dir {labels_dir} --output_dir {output_dir}"
                )
                if val_images_dir:
                    print(f"    --val_images_dir {val_images_dir} --val_labels_dir {val_labels_dir}")
                print()


if __name__ == "__main__":
    main()
