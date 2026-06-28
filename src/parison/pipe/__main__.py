from __future__ import annotations

import argparse
from typing import Sequence

from parison.pipe import evaluate, export, infer, train


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="parison", description="ICP digital-twin bench")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="anneal the twin on a recipe")
    p_train.add_argument("--recipe", required=True)
    p_train.add_argument("--out", default="runs/main")
    p_train.add_argument("--epochs", type=int, default=0)
    p_train.add_argument("--set", dest="overrides", nargs="*", default=[])

    p_eval = sub.add_parser("evaluate", help="score a checkpoint")
    p_eval.add_argument("--recipe", required=True)
    p_eval.add_argument("--ckpt", required=True)
    p_eval.add_argument("--set", dest="overrides", nargs="*", default=[])

    p_infer = sub.add_parser("infer", help="rank interventions for a batch")
    p_infer.add_argument("--recipe", required=True)
    p_infer.add_argument("--ckpt", required=True)
    p_infer.add_argument("--set", dest="overrides", nargs="*", default=[])

    p_export = sub.add_parser("export", help="serialize an ONNX graph")
    p_export.add_argument("--recipe", required=True)
    p_export.add_argument("--out", default="parison.onnx")
    p_export.add_argument("--set", dest="overrides", nargs="*", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    overrides: list[str] = list(args.overrides)
    if args.command == "train":
        train.run(args.recipe, args.out, overrides, args.epochs)
    elif args.command == "evaluate":
        evaluate.run(args.recipe, args.ckpt, overrides)
    elif args.command == "infer":
        infer.run(args.recipe, args.ckpt, overrides)
    elif args.command == "export":
        export.run(args.recipe, args.out, overrides)


if __name__ == "__main__":
    main()
