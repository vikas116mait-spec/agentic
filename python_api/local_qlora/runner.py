from __future__ import annotations

import argparse

from python_api.local_qlora.config import LocalQLoraJobConfig
from python_api.local_qlora.train import run_local_qlora_training


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a local GPU QLoRA training job.")
    parser.add_argument("config_path", help="Path to the local QLoRA job config JSON file.")
    args = parser.parse_args()

    config = LocalQLoraJobConfig.from_file(args.config_path)
    run_local_qlora_training(config)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
