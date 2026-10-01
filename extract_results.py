#!/usr/bin/env python3
"""
按 (dataset, seed) 分组，从 logs/<dataset>/*.txt 或 *.log 中提取评估指标。

文件名格式约定:
    <dataset>_seed<seed>_<YYYYMMDD>_<HHMMSS>.txt|.log
    例: ibm_l_medium_seed114514_20261001_145328.txt

用法:
    python extract_results.py logs
    python extract_results.py logs --raw           # 打印每个 seed 的原始数值
    python extract_results.py logs --csv results.csv
    python extract_results.py logs --verbose
    python extract_results.py logs --which ref     # 只输出 ref（refined）
    python extract_results.py logs --which orig    # 只输出 orig（extractor-only）
"""
import argparse
import csv
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

# 匹配 "Precision: 0.6575 | Recall: 0.6603 | F1: 0.5649 | Jaccard: 0.4413"
METRIC_RE = re.compile(
    r"Precision:\s*([\d.]+)\s*\|\s*Recall:\s*([\d.]+)\s*\|"
    r"\s*F1:\s*([\d.]+)\s*\|\s*Jaccard:\s*([\d.]+)"
)

# 解析文件名
FNAME_RE = re.compile(
    r"^(?P<dataset>.+?)_seed(?P<seed>\d+)_(?P<date>\d{8})_(?P<time>\d{6})\.(txt|log)$"
)


def parse_filename(path: Path):
    m = FNAME_RE.match(path.name)
    if not m:
        return None
    return m.group("dataset"), int(m.group("seed"))


def extract_metrics(path: Path):
    """返回 (orig, ref)。找不到两组就返回 None。"""
    text = path.read_text(encoding="utf-8", errors="ignore")
    matches = METRIC_RE.findall(text)
    if len(matches) < 2:
        return None
    orig = [float(x) for x in matches[-2]]
    ref = [float(x) for x in matches[-1]]
    return orig, ref


def cell(mean, std, n):
    if n <= 1:
        return f"{mean:.4f}"
    return f"{mean:.4f}±{std:.4f}"


def summarize(per_seed, key):
    """per_seed[ds][seed] = {"orig": [...], "ref": [...]}. key 是 'orig' 或 'ref'."""
    rows = []
    for ds in sorted(per_seed):
        seeds = sorted(per_seed[ds])
        if not seeds:
            continue
        P = [per_seed[ds][s][key][0] for s in seeds]
        R = [per_seed[ds][s][key][1] for s in seeds]
        F1 = [per_seed[ds][s][key][2] for s in seeds]
        J = [per_seed[ds][s][key][3] for s in seeds]
        n = len(seeds)

        def agg(x):
            m = statistics.mean(x)
            sd = statistics.stdev(x) if n > 1 else 0.0
            return m, sd

        Pm, Ps = agg(P)
        Rm, Rs = agg(R)
        Fm, Fs = agg(F1)
        Jm, Js = agg(J)
        rows.append({
            "dataset": ds,
            "n_seeds": n,
            "seeds": ",".join(str(s) for s in seeds),
            "P_mean": Pm, "P_std": Ps,
            "R_mean": Rm, "R_std": Rs,
            "F1_mean": Fm, "F1_std": Fs,
            "J_mean": Jm, "J_std": Js,
        })
    return rows


def print_table(rows, title):
    print()
    print("=" * 110)
    print(title)
    print("=" * 110)
    header = (f"{'dataset':<16} {'N':>3}  "
              f"{'Precision':>18}  {'Recall':>18}  {'F1':>18}  {'Jaccard':>18}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['dataset']:<16} {r['n_seeds']:>3}  "
              f"{cell(r['P_mean'],  r['P_std'],  r['n_seeds']):>18}  "
              f"{cell(r['R_mean'],  r['R_std'],  r['n_seeds']):>18}  "
              f"{cell(r['F1_mean'], r['F1_std'], r['n_seeds']):>18}  "
              f"{cell(r['J_mean'],  r['J_std'],  r['n_seeds']):>18}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs_dir", help="logs 根目录，例如 logs")
    ap.add_argument("--csv", help="把两张表都写到同一个 CSV")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--raw", action="store_true", help="打印每个 (dataset, seed) 的原始数值")
    ap.add_argument("--which", choices=["orig", "ref", "both"], default="both",
                    help="输出哪张表：orig=extractor-only，ref=+refiner（默认 both）")
    args = ap.parse_args()

    root = Path(args.logs_dir)
    if not root.is_dir():
        sys.exit(f"不是目录: {root}")

    per_seed = defaultdict(dict)

    for f in sorted(list(root.rglob("*.txt")) + list(root.rglob("*.log"))):
        parsed = parse_filename(f)
        if parsed is None:
            if args.verbose:
                print(f"[skip-name] {f}")
            continue
        ds, seed = parsed

        metrics = extract_metrics(f)
        if metrics is None:
            if args.verbose:
                print(f"[skip-metric] {f}")
            continue

        orig, ref = metrics
        if seed in per_seed[ds]:
            prev = per_seed[ds][seed]["file"]
            if f.name <= prev:
                continue
        per_seed[ds][seed] = {"orig": orig, "ref": ref, "file": f.name}
        if args.verbose:
            print(f"[ok] {ds}/seed={seed}  orig={orig}  ref={ref}  ({f.name})")

    rows_orig = summarize(per_seed, "orig")
    rows_ref = summarize(per_seed, "ref")

    if args.raw:
        for ds in sorted(per_seed):
            print(f"\n--- {ds} ---")
            for s in sorted(per_seed[ds]):
                o = per_seed[ds][s]["orig"]
                r = per_seed[ds][s]["ref"]
                print(f"  seed={s:<10} "
                      f"[orig] P={o[0]:.4f} R={o[1]:.4f} F1={o[2]:.4f} J={o[3]:.4f}   "
                      f"[ref] P={r[0]:.4f} R={r[1]:.4f} F1={r[2]:.4f} J={r[3]:.4f}")

    if args.which in ("orig", "both"):
        print_table(rows_orig,
                    "Extractor Only (without refinement) — grouped by (dataset, seed)")
    if args.which in ("ref", "both"):
        print_table(rows_ref,
                    "Full pipeline (+ Refiner) — grouped by (dataset, seed)")

    # 顺便打印差值（仅当两张表都存在时）
    if args.which == "both":
        print()
        print("=" * 110)
        print("Δ (Refiner - Extractor Only)  —  mean F1 / Jaccard")
        print("=" * 110)
        by_ds_o = {r["dataset"]: r for r in rows_orig}
        by_ds_r = {r["dataset"]: r for r in rows_ref}
        print(f"{'dataset':<16} {'ΔF1':>12} {'ΔJaccard':>12}")
        print("-" * 44)
        for ds in sorted(by_ds_r):
            if ds not in by_ds_o:
                continue
            dF = by_ds_r[ds]["F1_mean"] - by_ds_o[ds]["F1_mean"]
            dJ = by_ds_r[ds]["J_mean"] - by_ds_o[ds]["J_mean"]
            print(f"{ds:<16} {dF:>+12.4f} {dJ:>+12.4f}")

    if args.csv:
        with open(args.csv, "w", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(["# Extractor Only (without refinement)"])
            if rows_orig:
                w.writerow(list(rows_orig[0].keys()))
                for r in rows_orig:
                    w.writerow([r[k] for k in rows_orig[0].keys()])
            w.writerow([])
            w.writerow(["# Full pipeline (+ Refiner)"])
            if rows_ref:
                w.writerow(list(rows_ref[0].keys()))
                for r in rows_ref:
                    w.writerow([r[k] for k in rows_ref[0].keys()])
        print(f"\n已写到 {args.csv}")


if __name__ == "__main__":
    main()