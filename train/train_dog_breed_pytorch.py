from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from train.common_pytorch import build_arg_parser, train_model  # noqa: E402


def main() -> None:
    args = build_arg_parser("dog_breed", default_epochs=40).parse_args()
    train_model("dog_breed", args)


if __name__ == "__main__":
    main()
