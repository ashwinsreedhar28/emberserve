"""Matplotlib figures from the benchmark JSON files.

    python -m pagedserve.bench.plot results/vllm.json results/pagedserve.json \
        --ablation results/ablation.json --out-dir results/plots

Sweep files (from run_vllm_baseline) hold `runs[]` keyed by request_rate; the
ablation file holds `results[]` keyed by config.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Categorical hues in fixed order (colorblind-validated set); one system per slot.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
GRID = "#e4e3df"
TEXT = "#0b0b0b"
MUTED = "#52514e"
DPI = 150

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
})


def _rate_label(rate: float | None) -> str:
    return "inf" if rate is None else f"{rate:g}"


def _rates_x(runs: list[dict]) -> tuple[list[float], list[str]]:
    """X positions for a rate sweep: finite rates keep their value; 'inf' sits one step
    beyond the largest finite rate so it stays on a log axis."""
    finite = [r["request_rate"] for r in runs if r["request_rate"] is not None]
    inf_x = (max(finite) * 2) if finite else 1.0
    xs = [inf_x if r["request_rate"] is None else r["request_rate"] for r in runs]
    return xs, [_rate_label(r["request_rate"]) for r in runs]


def _sweep_axes(ax: plt.Axes, runs: list[dict], ylabel: str) -> None:
    xs, labels = _rates_x(runs)
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.minorticks_off()
    ax.set_xlabel("offered request rate (req/s)")
    ax.set_ylabel(ylabel)


def plot_throughput_vs_rate(sweeps: dict[str, dict], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for i, (name, data) in enumerate(sweeps.items()):
        runs = data["runs"]
        xs, _ = _rates_x(runs)
        ys = [r["summary"]["throughput_tok_s"] for r in runs]
        ax.plot(xs, ys, marker="o", ms=4, lw=2, color=SERIES[i % len(SERIES)], label=name)
        ax.annotate(name, (xs[-1], ys[-1]), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8, color=TEXT)
    _sweep_axes(ax, next(iter(sweeps.values()))["runs"], "output throughput (tok/s)")
    ax.set_ylim(bottom=0)
    ax.set_title("Output throughput vs offered load")
    if len(sweeps) >= 2:
        ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def _latency_vs_rate(sweeps: dict[str, dict], metric: str, title: str, ylabel: str,
                     out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for i, (name, data) in enumerate(sweeps.items()):
        runs = data["runs"]
        xs, _ = _rates_x(runs)
        color = SERIES[i % len(SERIES)]
        p50 = [r["summary"][metric]["p50"] for r in runs]
        p99 = [r["summary"][metric]["p99"] for r in runs]
        ax.plot(xs, p50, marker="o", ms=4, lw=2, color=color, label=f"{name} p50")
        ax.plot(xs, p99, marker="o", ms=4, lw=1.5, ls="--", color=color, label=f"{name} p99")
    _sweep_axes(ax, next(iter(sweeps.values()))["runs"], ylabel)
    ax.set_yscale("log")
    ax.set_title(title)
    ax.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def plot_ttft_vs_rate(sweeps: dict[str, dict], out: Path) -> Path:
    return _latency_vs_rate(sweeps, "ttft_ms", "Time to first token vs offered load",
                            "TTFT (ms, log)", out)


def plot_tpot_vs_rate(sweeps: dict[str, dict], out: Path) -> Path:
    return _latency_vs_rate(sweeps, "tpot_ms", "Time per output token vs offered load",
                            "TPOT (ms, log)", out)


def plot_ablation(ablation: dict, out: Path) -> Path:
    results = ablation["results"]
    names = [r["config"] for r in results]
    tok = [r["summary"]["throughput_tok_s"] for r in results]
    ttft = [r["summary"]["ttft_ms"]["p50"] for r in results]
    kv = [100 * r.get("kv_utilization_mean", 0.0) for r in results]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    for ax, vals, ylabel, title in zip(
            axes, (tok, ttft, kv),
            ("output throughput (tok/s)", "TTFT p50 (ms)", "mean KV utilization (%)"),
            ("Throughput", "TTFT p50", "KV cache utilization"), strict=True):
        bars = ax.bar(range(len(names)), vals, width=0.62, color=SERIES[0], edgecolor="white",
                      linewidth=1)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
        for b, v in zip(bars, vals, strict=True):
            ax.annotate(f"{v:.0f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7,
                        color=MUTED)
    model = ablation.get("model", "")
    gpu = ablation.get("gpu") or ablation.get("device", "")
    fig.suptitle(f"Ablation: {model} on {gpu}  (n={ablation['trace']['n']}, "
                 f"rate={_rate_label(ablation['trace']['request_rate'])} req/s)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def plot_all(sweeps: dict[str, dict], ablation: dict | None, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    if sweeps:
        written.append(plot_throughput_vs_rate(sweeps, out_dir / "throughput_vs_rate.png"))
        written.append(plot_ttft_vs_rate(sweeps, out_dir / "ttft_vs_rate.png"))
        written.append(plot_tpot_vs_rate(sweeps, out_dir / "tpot_vs_rate.png"))
    if ablation is not None:
        written.append(plot_ablation(ablation, out_dir / "ablation.png"))
    return written


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="pagedserve.bench.plot", description=__doc__)
    p.add_argument("sweeps", nargs="*", help="sweep JSON files from run_vllm_baseline")
    p.add_argument("--ablation", default=None, help="ablation JSON from ablation.py")
    p.add_argument("--out-dir", default="results/plots")
    p.add_argument("--labels", default=None,
                   help="comma-separated legend labels, one per sweep file (default: the file's system name)")
    args = p.parse_args(argv)
    labels = args.labels.split(",") if args.labels else [None] * len(args.sweeps)
    if len(labels) != len(args.sweeps):
        raise SystemExit("--labels must have one entry per sweep file")
    sweeps: dict[str, Any] = {}
    for f, label in zip(args.sweeps, labels):
        data = json.loads(Path(f).read_text())
        sweeps[label or data.get("system") or Path(f).stem] = data
    ablation = json.loads(Path(args.ablation).read_text()) if args.ablation else None
    for path in plot_all(sweeps, ablation, Path(args.out_dir)):
        print(f"wrote {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
