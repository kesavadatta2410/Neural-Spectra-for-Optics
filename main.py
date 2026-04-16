"""
SpecCompress-India: Main CLI  (v2)
Routes: generate | train | eval | ablation | compression | temp | llm | duke
"""

import argparse, subprocess, sys, os


def run(cmd):
    print("[main]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser(description="SpecCompress-India v2")
    s = p.add_subparsers(dest="cmd", required=True)

    # generate
    g = s.add_parser("generate")
    g.add_argument("--n_train",  type=int, default=50000)
    g.add_argument("--n_val",    type=int, default=5000)
    g.add_argument("--n_test",   type=int, default=5000)
    g.add_argument("--n_points", type=int, default=1000)
    g.add_argument("--out_dir",  default="data/synthetic")
    g.add_argument("--seed",     type=int, default=42)

    # train
    t = s.add_parser("train")
    t.add_argument("--config",     default="configs/config.yaml")
    t.add_argument("--epochs",     type=int, default=None)
    t.add_argument("--batch_size", type=int, default=None)

    # eval — full suite (all gaps)
    e = s.add_parser("eval")
    e.add_argument("--config",  default="configs/config.yaml")
    e.add_argument("--ckpt",    default=None)
    e.add_argument("--api_key", default=None, help="Anthropic key for Gap 2")
    e.add_argument("--skip_gaps", nargs="*", default=[])

    # individual gap runners
    s.add_parser("ablation").add_argument("--config", default="configs/config.yaml")
    s.add_parser("compression").add_argument("--config", default="configs/config.yaml")
    cr = s.add_parser("temp")
    cr.add_argument("--config", default="configs/config.yaml")
    cr.add_argument("--ckpt",   default="experiments/checkpoints/best.pt")
    l = s.add_parser("llm")
    l.add_argument("--config",  default="configs/config.yaml")
    l.add_argument("--api_key", default=None)

    args = p.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    py = sys.executable

    if args.cmd == "generate":
        run([py, "data/data_generation.py",
             "--n_train", str(args.n_train), "--n_val", str(args.n_val),
             "--n_test",  str(args.n_test),  "--n_points", str(args.n_points),
             "--out_dir", args.out_dir,       "--seed", str(args.seed)])

    elif args.cmd == "train":
        cmd = [py, "training/train.py", "--config", args.config]
        if args.epochs:     cmd += ["--epochs", str(args.epochs)]
        if args.batch_size: cmd += ["--batch_size", str(args.batch_size)]
        run(cmd)

    elif args.cmd == "eval":
        cmd = [py, "evaluation/eval.py", "--config", args.config]
        if args.ckpt:    cmd += ["--ckpt", args.ckpt]
        if args.api_key: cmd += ["--api_key", args.api_key]
        if args.skip_gaps: cmd += ["--skip_gaps"] + args.skip_gaps
        run(cmd)

    elif args.cmd == "ablation":
        run([py, "evaluation/ablation.py", "--config", args.config])

    elif args.cmd == "compression":
        run([py, "evaluation/compression_ablation.py", "--config", args.config])

    elif args.cmd == "temp":
        run([py, "evaluation/temp_robustness.py",
             "--config", args.config, "--ckpt", args.ckpt])

    elif args.cmd == "llm":
        cmd = [py, "-c",
               f"import sys; sys.path.insert(0,'.'); "
               f"from utils.llm_integration import run_llm_benchmark, print_llm_results; "
               f"print('Use: python evaluation/eval.py --skip_gaps 1 3 4 5 6')"]
        run(cmd)


if __name__ == "__main__":
    main()
