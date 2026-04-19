#!/usr/bin/env python3
"""
Phase 2 Results Parser
Run: python3 phase2_parse.py
Reads ~/results/phase2/*.json, prints tuning table, saves chart.
"""

import json, glob, re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS_DIR = Path.home() / "results" / "phase2"

COLORS = {"bfq": "#0d6efd", "kyber": "#198754", "mq-deadline": "#dc3545"}

# Parameter groups for display
PARAM_GROUPS = {
    "BFQ":        ["bfq_slice_idle", "bfq_low_latency", "bfq_timeout_sync"],
    "Kyber":      ["kyber_read_lat", "kyber_write_lat"],
    "MQ-Deadline":["mqd_read_expire", "mqd_write_expire", "mqd_writes_starved"],
}

# ── Parse ─────────────────────────────────────────────────────────────────────

def parse(filepath):
    data = json.load(open(filepath))
    jobs = data.get("jobs", [])
    if not jobs:
        return None
    # Grab foreground (fg) read job
    job = next((j for j in jobs if "fg" in j.get("jobname", "")), jobs[0])
    read = job.get("read", {})
    iops = read.get("iops", 0)
    pcts = read.get("clat_ns", {}).get("percentile", {})
    p99 = next((v/1000 for k,v in pcts.items() if abs(float(k)-99.0)<0.01), None)
    return {"iops": iops, "p99_us": p99}

records = []
for path in sorted(RESULTS_DIR.glob("*.json")):
    m = parse(path)
    if not m:
        continue
    name = path.stem  # e.g. "bfq_slice_idle_8"
    # Detect scheduler from filename prefix
    if name.startswith("bfq"):
        sched = "bfq"
    elif name.startswith("kyber"):
        sched = "kyber"
    else:
        sched = "mq-deadline"
    records.append({"label": name, "scheduler": sched, **m})

df = pd.DataFrame(records)

# ── Print table ───────────────────────────────────────────────────────────────

# Phase 1 baseline P99 for reference
PHASE1_BASELINE = {
    "bfq": 69730.3,
    "kyber": 1233125.4,
    "mq-deadline": 994050.0,
}

print("\n" + "="*70)
print("  PHASE 2: PARAMETER TUNING RESULTS (W2 Interference Workload)")
print("  Reference: Phase 1 default P99 shown for comparison")
print("="*70)

for group, prefixes in PARAM_GROUPS.items():
    print(f"\n── {group} ──")
    print(f"  {'Config':<35} {'IOPS':>6} {'P99 (µs)':>12} {'vs Default':>12}")
    print(f"  {'-'*35} {'-'*6} {'-'*12} {'-'*12}")

    sched_key = group.lower().replace("-", "").replace(" ", "")
    sched_key = "bfq" if group=="BFQ" else "kyber" if group=="Kyber" else "mq-deadline"
    baseline_p99 = PHASE1_BASELINE[sched_key]

    group_df = df[df["label"].str.startswith(tuple(prefixes))]
    best_p99 = group_df["p99_us"].min() if not group_df.empty else None

    for _, row in group_df.iterrows():
        p99 = row["p99_us"]
        pct_change = ((p99 - baseline_p99) / baseline_p99 * 100) if p99 and baseline_p99 else 0
        star = " ★" if p99 == best_p99 else "  "
        change_str = f"{pct_change:+.1f}%" if p99 else "N/A"
        p99_str = f"{p99:>12.0f}" if p99 else f"{'N/A':>12}"
        print(f"  {row['label']:<35} {row['iops']:>6.0f} {p99_str} {change_str:>12}{star}")

print("\n  ★ = best P99 in group | vs Default = change from Phase 1 baseline")
print("="*70 + "\n")

# ── Best configs summary ──────────────────────────────────────────────────────

print("  BEST CONFIGS PER SCHEDULER (use these as ML training labels):")
print(f"  {'-'*60}")
for sched in ["bfq", "kyber", "mq-deadline"]:
    sdf = df[df.scheduler == sched]
    if sdf.empty:
        continue
    best = sdf.loc[sdf.p99_us.idxmin()]
    print(f"  {sched:<14} best config: {best['label']:<30} P99={best['p99_us']:.0f}µs")
print()

# ── Chart ─────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Phase 2: P99 Latency Under Parameter Tuning (W2 Interference)",
             fontsize=12, fontweight="bold")

for ax, (group, prefixes) in zip(axes, PARAM_GROUPS.items()):
    sched = "bfq" if group=="BFQ" else "kyber" if group=="Kyber" else "mq-deadline"
    gdf = df[df["label"].str.startswith(tuple(prefixes))].copy()

    if gdf.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_title(group)
        continue

    # Short labels for x-axis
    gdf["short"] = gdf["label"].str.replace(r"^(bfq_|kyber_|mqd_)", "", regex=True)
    vals = gdf["p99_us"].values
    labels = gdf["short"].values

    bars = ax.bar(range(len(vals)), vals, color=COLORS[sched], alpha=0.8, edgecolor="white")

    # Mark best
    best_i = int(np.argmin(vals))
    bars[best_i].set_edgecolor("gold")
    bars[best_i].set_linewidth(2.5)

    # Phase 1 baseline reference line
    ax.axhline(PHASE1_BASELINE[sched], color="black", linestyle="--",
               linewidth=1, alpha=0.5, label="Phase 1 default")

    ax.set_title(f"{group}", fontsize=10)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("P99 Latency (µs)" if ax == axes[0] else "")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

plt.tight_layout()
out = Path.home() / "results" / "phase2_tuning.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Chart saved: {out}")
plt.show()
