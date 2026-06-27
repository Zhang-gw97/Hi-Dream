"""Training entry point for Hi-DREAM.

This file currently provides the command-line interface and configuration loading
logic. The full training pipeline will be added in the public code release.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Hi-DREAM model.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a YAML configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    project_name = config.get("project", {}).get("name", "Hi-DREAM")
    output_dir = config.get("training", {}).get("output_dir", "outputs")

    print(f"Loaded configuration for {project_name}")
    print(f"Output directory: {output_dir}")
    print("Training implementation will be added in the full release.")


if __name__ == "__main__":
    main()
