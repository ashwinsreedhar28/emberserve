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


def collapse_repeats(data: dict) -> dict:
    """A sweep that ran a rate several times (`--rates inf,inf,inf`) becomes one run per
    rate with the summary's numbers averaged (throughput, and the mean/p50/p90/p99 of each
    latency), so the figures show the mean of the repeats rather than three points."""
    groups: dict = {}
    order: list = []
    for r in data["runs"]:
        key = r["request_rate"]
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(r)
    if all(len(g) == 1 for g in groups.values()):
        return data
    runs = []
    for key in order:
        g = groups[key]
        base = json.loads(json.dumps(g[0]))
        s = base["summary"]
        n = len(g)
        s["throughput_tok_s"] = sum(r["summary"]["throughput_tok_s"] for r in g) / n
        for metric in ("ttft_ms", "tpot_ms", "e2e_ms"):
            if isinstance(s.get(metric), dict):
                for q in s[metric]:
                    s[metric][q] = sum(r["summary"][metric][q] for r in g) / n
        base["repeats"] = n
        runs.append(base)
    out = dict(data)
    out["runs"] = runs
    return out


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


# ---- the fix-by-fix progression: every version of one system against the baseline -------
def _progression_colors(n: int) -> list[str]:
    """A sequential ramp for the versions (oldest lightest), so the eye reads the order."""
    cmap = matplotlib.colormaps["Blues"]
    if n == 1:
        return [matplotlib.colors.to_hex(cmap(0.95))]
    return [matplotlib.colors.to_hex(cmap(0.35 + 0.6 * i / (n - 1))) for i in range(n)]


def _progression_axes(ax: plt.Axes, baseline: dict | None, versions: dict[str, dict],
                      value, ylabel: str, title: str, log: bool, end_labels: bool = False) -> None:
    colors = _progression_colors(len(versions))
    ref = baseline if baseline is not None else next(iter(versions.values()))
    ends: list[tuple[float, float, str, str]] = []
    for (name, data), color in zip(versions.items(), colors, strict=True):
        runs = data["runs"]
        xs, _ = _rates_x(runs)
        ys = [value(r) for r in runs]
        ax.plot(xs, ys, marker="o", ms=3.5, lw=1.8, color=color, label=name)
        ends.append((xs[-1], ys[-1], name.split(" ")[0], color))
    if baseline is not None:
        runs = baseline["runs"]
        xs, _ = _rates_x(runs)
        ys = [value(r) for r in runs]
        ax.plot(xs, ys, marker="s", ms=4, lw=2.4, ls="--", color=SERIES[1], label="vLLM")
        ends.append((xs[-1], ys[-1], "vLLM", SERIES[1]))
    if end_labels:  # the saturation values, nudged apart so neighbours stay legible
        ends.sort(key=lambda e: e[1])
        span = (ax.get_ylim()[1] - ax.get_ylim()[0]) or 1.0
        last_y = None
        for x, y, tag, color in ends:
            ty = y if last_y is None else max(y, last_y + 0.035 * span)
            ax.annotate(f"{tag} {y:,.0f}", (x, y), xytext=(6, (ty - y) / span * 300),
                        textcoords="offset points", va="center", fontsize=7, color=color)
            last_y = ty
    _sweep_axes(ax, ref["runs"], ylabel)
    if log:
        ax.set_yscale("log")
    else:
        ax.set_ylim(bottom=0)
    ax.set_title(title)
    ax.legend(loc="upper left", ncol=2)


def plot_progression(baseline: dict | None, versions: dict[str, dict], out_dir: Path,
                     system: str = "pagedserve") -> list[Path]:
    """Three figures with one line per version: throughput, TPOT p50, TPOT p99 (and TTFT
    p50), the baseline dashed. Versions that were only swept at the high rates start there."""
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("progression_throughput.png", lambda r: r["summary"]["throughput_tok_s"],
         "output throughput (tok/s)", f"{system}: throughput vs offered load, by version", False),
        ("progression_tpot_p50.png", lambda r: r["summary"]["tpot_ms"]["p50"],
         "TPOT p50 (ms, log)", f"{system}: time per output token (p50), by version", True),
        ("progression_tpot_p99.png", lambda r: r["summary"]["tpot_ms"]["p99"],
         "TPOT p99 (ms, log)", f"{system}: time per output token (p99), by version", True),
        ("progression_ttft_p50.png", lambda r: r["summary"]["ttft_ms"]["p50"],
         "TTFT p50 (ms, log)", f"{system}: time to first token (p50), by version", True),
    ]
    written = []
    for fname, value, ylabel, title, log in specs:
        fig, ax = plt.subplots(figsize=(6.8, 4.0))
        _progression_axes(ax, baseline, versions, value, ylabel, title, log, end_labels=not log)
        if not log:
            ax.set_xlim(right=ax.get_xlim()[1] * 1.6)  # room for the end labels
        fig.tight_layout()
        out = out_dir / fname
        fig.savefig(out, dpi=DPI)
        plt.close(fig)
        written.append(out)
    return written


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
    p.add_argument("--progression", default=None, metavar="BASELINE_JSON",
                   help="fix-by-fix figures: the positional sweeps are the versions, in order, "
                        "of one system; this file is the baseline drawn dashed")
    args = p.parse_args(argv)
    labels = args.labels.split(",") if args.labels else [None] * len(args.sweeps)
    if len(labels) != len(args.sweeps):
        raise SystemExit("--labels must have one entry per sweep file")
    sweeps: dict[str, Any] = {}
    for f, label in zip(args.sweeps, labels):
        data = collapse_repeats(json.loads(Path(f).read_text()))
        sweeps[label or data.get("system") or Path(f).stem] = data
    if args.progression:
        baseline = collapse_repeats(json.loads(Path(args.progression).read_text()))
        for path in plot_progression(baseline, sweeps, Path(args.out_dir)):
            print(f"wrote {path}", file=sys.stderr)
        return 0
    ablation = json.loads(Path(args.ablation).read_text()) if args.ablation else None
    for path in plot_all(sweeps, ablation, Path(args.out_dir)):
        print(f"wrote {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
