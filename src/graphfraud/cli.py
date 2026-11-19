"""Command-line entry point: ``graphfraud describe`` and ``graphfraud run-all``."""

from __future__ import annotations

import argparse
import json

from .data import load_elliptic
from .gnn import GNNConfig
from .pipeline import MODELS, RunConfig, run_all
from .split import TemporalSplit


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="graphfraud", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("describe", help="print dataset statistics")
    d.add_argument("--raw-dir", default="data/raw")

    r = sub.add_parser("run-all", help="train every model and write metrics + figures")
    r.add_argument("--raw-dir", default="data/raw")
    r.add_argument("--out-dir", default="docs/results")
    r.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    r.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    r.add_argument("--val-start", type=int, default=30)
    r.add_argument("--train-end", type=int, default=34)
    r.add_argument("--dark-market-step", type=int, default=43)
    r.add_argument("--cost-missed", type=float, default=10.0)
    r.add_argument("--cost-false-alert", type=float, default=1.0)
    r.add_argument("--max-epochs", type=int, default=200)
    r.add_argument("--n-boot", type=int, default=2000)
    r.add_argument("--n-jobs", type=int, default=2, help="threads (results depend on it)")
    r.add_argument(
        "--no-rolling-threshold", action="store_true",
        help="skip the rolling-origin refits used to pick a threshold for the deployed model",
    )

    args = parser.parse_args(argv)
    if args.cmd == "describe":
        print(json.dumps(load_elliptic(args.raw_dir).summary(), indent=2))
        return
    cfg = RunConfig(
        seeds=tuple(args.seeds),
        c_fn=args.cost_missed,
        c_fp=args.cost_false_alert,
        split=TemporalSplit(args.val_start, args.train_end, args.dark_market_step),
        gnn=GNNConfig(max_epochs=args.max_epochs),
        n_boot=args.n_boot,
        models=tuple(args.models),
        rolling_threshold=not args.no_rolling_threshold,
        n_jobs=args.n_jobs,
    )
    result = run_all(args.raw_dir, args.out_dir, cfg)
    for name, m in result["models"].items():
        s = m["summary"]
        print(
            f"{name:14s} F1 {s['f1']['mean']:.3f} +/- {s['f1']['sd']:.3f}  "
            f"AUPRC {s['auprc']['mean']:.3f}  "
            f"post-shutdown F1 {s['f1_post_shutdown']['mean']:.3f}  "
            f"cost@0.5 {s['cost_at_0.5']['mean']:.4f}  "
            + "  ".join(
                f"{k} {v['cost']['mean']:.4f}"
                for k, v in s["threshold_variants"].items()
                if "cost" in v
            )
        )


if __name__ == "__main__":
    main()
