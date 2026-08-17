from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from train.common_tensorflow import build_arg_parser, train_model  # noqa: E402


def main() -> None:
    args = build_arg_parser("dog_breed", default_epochs=80, default_head="gap", default_aug="strong", default_mixup=0.2, default_patience=12).parse_args()
    train_model("dog_breed", args)


if __name__ == "__main__":
    main()
