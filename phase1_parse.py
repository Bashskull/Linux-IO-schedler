#!/usr/bin/env python3
"""
Phase 1 Results Parser
Run: python3 phase1_parse.py
Reads JSON files from ~/results/, prints baseline table, saves 2 charts.
"""

import json, glob
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path.home() / "results"
SCHEDULERS  = ["none", "bfq", "kyber", "mq-deadline"]
WORKLOADS   = {"W1_latency": "W1: Latency (randread 4K QD=1)",
               "W2_interference": "W2: Interference (fg reads + bg writes)"}
COLORS      = {"none": "#888", "bfq": "#0d6efd", "kyber": "#198754", "mq-deadline": "#dc3545"}

# ── Parse ─────────────────────────────────────────────────────────────────────

def parse(filepath):
    data = json.load(open(filepath))
    jobs = data.get("jobs", [])
    if not jobs:
        return None

    # For W2 (interference), we care about the foreground read job's latency
    job = next((j for j in jobs if "fg" in j.get("jobname","") or len(jobs)==1), jobs[0])
    read = job.get("read", {})

    iops = read.get("iops", 0)
    pcts = read.get("clat_ns", {}).get("percentile", {})

    p99 = next((v/1000 for k,v in pcts.items() if abs(float(k)-99.0)<0.01), None)
    p50 = next((v/1000 for k,v in pcts.items() if abs(float(k)-50.0)<0.01), None)

    return {"iops": iops, "p50_us": p50, "p99_us": p99}

records = []
for wkey in WORKLOADS:
    for sched in SCHEDULERS:
        path = RESULTS_DIR / f"{wkey}_{sched}.json"
        if not path.exists():
            print(f"Missing: {path.name} — skipping")
            continue
        m = parse(path)
        if m:
            records.append({"workload": wkey, "scheduler": sched, **m})

df = pd.DataFrame(records)

# ── Print table ───────────────────────────────────────────────────────────────

print("\n" + "="*65)
print("  PHASE 1 BASELINE RESULTS")
print("="*65)

for wkey, wlabel in WORKLOADS.items():
    wdf = df[df.workload == wkey]
    print(f"\n  {wlabel}")
    print(f"  {'Scheduler':<14} {'IOPS':>8} {'P50 (µs)':>10} {'P99 (µs)':>10}")
    print(f"  {'-'*14} {'-'*8} {'-'*10} {'-'*10}")

    best_p99 = wdf.p99_us.min()
    for _, row in wdf.iterrows():
        star = " ★" if row.p99_us == best_p99 else "  "
        print(f"  {row.scheduler:<14} {row.iops:>8.0f} {row.p50_us:>10.1f} {row.p99_us:>10.1f}{star}")

print("\n  ★ = lowest P99 (best latency) for this workload")
print("="*65 + "\n")

# ── Charts ────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Phase 1: P99 Tail Latency by Scheduler", fontsize=13, fontweight="bold")

for ax, (wkey, wlabel) in zip(axes, WORKLOADS.items()):
    wdf = df[df.workload == wkey]
    vals   = [wdf[wdf.scheduler==s]["p99_us"].values[0] if s in wdf.scheduler.values else 0 for s in SCHEDULERS]
    colors = [COLORS[s] for s in SCHEDULERS]
    bars   = ax.bar(SCHEDULERS, vals, color=colors, alpha=0.85, edgecolor="white")
    # Gold border on best
    best_i = int(np.argmin(vals))
    bars[best_i].set_edgecolor("gold"); bars[best_i].set_linewidth(2.5)

    ax.set_title(wlabel, fontsize=10)
    ax.set_ylabel("P99 Latency (µs)")
    ax.set_xlabel("Scheduler")
    ax.grid(axis="y", alpha=0.3)
    for spine in ["top","right"]: ax.spines[spine].set_visible(False)

plt.tight_layout()
out = RESULTS_DIR / "phase1_p99.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Chart saved: {out}")
plt.show()
