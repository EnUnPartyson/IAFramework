from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from train.common_pytorch import build_arg_parser, train_model  # noqa: E402
from utils.model_defs_pytorch import DETECTOR_CLASS_NAMES  # noqa: E402


def main() -> None:
    args = build_arg_parser("detector").parse_args()
    train_model("detector", args, expected_classes=DETECTOR_CLASS_NAMES)


if __name__ == "__main__":
    main()
