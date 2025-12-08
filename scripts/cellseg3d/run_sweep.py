#!/usr/bin/env python3
"""Launch a WandB hyperparameter sweep for SwinUNETR training."""

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


def parse_depths_string(depths_str: str) -> list:
    """Parse a string like '[2, 2, 2, 2]' into a list of integers."""
    try:
        depths = ast.literal_eval(depths_str)
        if isinstance(depths, (list, tuple)) and len(depths) == 4:
            return list(depths)
        raise ValueError("depths must be a list/tuple of 4 integers")
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"Invalid depths format '{depths_str}': {e}")


def main():
    parser = argparse.ArgumentParser(description="Launch WandB hyperparameter sweep")
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--labels_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--val_images_dir", type=str, default=None)
    parser.add_argument("--val_labels_dir", type=str, default=None)
    parser.add_argument("--sweep_config", type=str, default="sweep_config.yaml")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--run_agent", action="store_true")
    parser.add_argument("--sweep_id", type=str, default=None)
    parser.add_argument("--project", type=str, default="CellSeg3D")

    args = parser.parse_args()

    script_dir = Path(__file__).parent
    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    output_dir = Path(args.output_dir)
    sweep_config_path = script_dir / args.sweep_config

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
    if not sweep_config_path.exists():
        raise ValueError(f"Sweep config file does not exist: {sweep_config_path}")

    val_images_dir = Path(args.val_images_dir) if args.val_images_dir else None
    val_labels_dir = Path(args.val_labels_dir) if args.val_labels_dir else None

    if (val_images_dir is not None) ^ (val_labels_dir is not None):
        raise ValueError("Both --val_images_dir and --val_labels_dir must be provided together")

    with open(sweep_config_path, "r") as f:
        sweep_config = yaml.safe_load(f)

    # Handle depths parameter: convert string format to list
    if "parameters" in sweep_config and "depths" in sweep_config["parameters"]:
        depths_values = sweep_config["parameters"]["depths"]["values"]
        parsed_depths = [
            parse_depths_string(d) if isinstance(d, str) else d for d in depths_values
        ]
        sweep_config["parameters"]["depths"]["values"] = parsed_depths

    def train():
        cmd = [
            sys.executable,
            str(script_dir / "train_cellseg3d_swinunetr.py"),
            "--images_dir",
            str(images_dir),
            "--labels_dir",
            str(labels_dir),
            "--output_dir",
            str(output_dir),
            "--device",
            args.device,
            "--max_epochs",
            str(args.max_epochs),
        ]
        if val_images_dir:
            cmd.extend(["--val_images_dir", str(val_images_dir)])
        if val_labels_dir:
            cmd.extend(["--val_labels_dir", str(val_labels_dir)])
        subprocess.run(cmd, check=True)

    if args.sweep_id is not None:
        print(f"Running agent for sweep: {args.sweep_id}")
        print("=" * 60)
        wandb.agent(
            sweep_id=args.sweep_id,
            function=train,
            project=args.project,
            entity=entity,
            count=args.count,
        )
    else:
        print("=" * 60)
        print("Initializing WandB Sweep")
        print("=" * 60)
        print(f"Project: {args.project}")
        print(f"Sweep config: {sweep_config_path}")
        print(f"Method: {sweep_config.get('method', 'grid')}")
        print("=" * 60)

        sweep_id = wandb.sweep(sweep_config, project=args.project, entity=entity)

        print(f"\nSweep initialized with ID: {sweep_id}")
        print(f"View at: https://wandb.ai/{entity}/sweeps/{sweep_id}")

        if args.run_agent:
            print("=" * 60)
            print("Starting agent...")
            print("=" * 60)
            wandb.agent(
                sweep_id=sweep_id,
                function=train,
                project=args.project,
                entity=entity,
                count=args.count,
            )
        else:
            print(f"\nTo run agents, use:")
            print(f"  wandb agent {args.project}/{sweep_id}")
            print(
                f"\nOr: python run_sweep.py --run_agent --sweep_id {sweep_id} "
                f"--images_dir {images_dir} --labels_dir {labels_dir} --output_dir {output_dir}"
            )

        print("=" * 60)
        print(f"Sweep ID: {sweep_id}")
        print("=" * 60)


if __name__ == "__main__":
    main()
