"""
SpecCompress-India: Main Entry Point
Routes to: generate | train | eval
Usage:
    python main.py generate
    python main.py train   [--config configs/config.yaml]
    python main.py eval    [--config configs/config.yaml]
"""

import argparse
import subprocess
import sys
import os


def cmd_generate(args):
    cmd = [
        sys.executable, "data/data_generation.py",
        "--n_train",  str(args.n_train),
        "--n_val",    str(args.n_val),
        "--n_test",   str(args.n_test),
        "--n_points", str(args.n_points),
        "--out_dir",  args.out_dir,
        "--seed",     str(args.seed),
    ]
    print("[main] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def cmd_train(args):
    cmd = [sys.executable, "training/train.py", "--config", args.config]
    if args.epochs:
        cmd += ["--epochs", str(args.epochs)]
    if args.batch_size:
        cmd += ["--batch_size", str(args.batch_size)]
    print("[main] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def cmd_eval(args):
    cmd = [sys.executable, "evaluation/eval.py", "--config", args.config]
    if args.ckpt:
        cmd += ["--ckpt", args.ckpt]
    print("[main] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="SpecCompress-India Pipeline")
    sub    = parser.add_subparsers(dest="command", required=True)

    # generate
    p_gen = sub.add_parser("generate", help="Generate synthetic datasets")
    p_gen.add_argument("--n_train",  type=int, default=50000)
    p_gen.add_argument("--n_val",    type=int, default=5000)
    p_gen.add_argument("--n_test",   type=int, default=5000)
    p_gen.add_argument("--n_points", type=int, default=1000)
    p_gen.add_argument("--out_dir",  type=str, default="data/synthetic")
    p_gen.add_argument("--seed",     type=int, default=42)

    # train
    p_train = sub.add_parser("train", help="Train SpecCompress model")
    p_train.add_argument("--config",     type=str, default="configs/config.yaml")
    p_train.add_argument("--epochs",     type=int, default=None)
    p_train.add_argument("--batch_size", type=int, default=None)

    # eval
    p_eval = sub.add_parser("eval", help="Evaluate and compare baselines")
    p_eval.add_argument("--config", type=str, default="configs/config.yaml")
    p_eval.add_argument("--ckpt",   type=str, default=None)

    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "eval":
        cmd_eval(args)


if __name__ == "__main__":
    main()
