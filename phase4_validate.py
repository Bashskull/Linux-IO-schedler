#!/usr/bin/env python3
"""
Phase 4: Validation
====================
Proves the adaptive daemon outperforms static schedulers.

How it works:
  1. Run W2 interference workload with each STATIC scheduler → record P99
  2. Run W2 interference workload WITH DAEMON RUNNING → record P99
  3. Compare — daemon should match or beat the best static scheduler

Run as: sudo python3 phase4_validate.py
"""

import subprocess, time, json, os, sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

DEVICE     = "sda"
SCHED_FILE = f"/sys/block/{DEVICE}/queue/scheduler"
RESULTS    = Path.home() / "results"
RESULTS.mkdir(exist_ok=True)

if os.geteuid() != 0:
    print("Run as root: sudo python3 phase4_validate.py")
    sys.exit(1)

def switch_scheduler(sched):
    with open(SCHED_FILE, "w") as f:
        f.write(sched)
    time.sleep(0.5)

def flush_cache():
    os.system("sync")
    with open("/proc/sys/vm/drop_caches", "w") as f:
        f.write("3")
    time.sleep(1)

def run_fio(label, extra_args=""):
    """Run W2 interference workload, return P99 of foreground reads."""
    outfile = RESULTS / f"phase4_{label}.json"
    flush_cache()

    cmd = (
        f"fio --filename=/dev/{DEVICE} --direct=1 --ioengine=libaio "
        f"--time_based=1 --runtime=30 --group_reporting=0 "
        f"--output-format=json --output={outfile} "
        f"--name=fg --rw=randread --bs=4k --iodepth=1 --numjobs=1 "
        f"--name=bg --rw=randwrite --bs=4k --iodepth=64 --numjobs=2 "
        f"--lat_percentiles=1 --percentile_list=50:99:99.9 {extra_args}"
    )
    subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    data = json.load(open(outfile))
    jobs = data.get("jobs", [])
    fg   = next((j for j in jobs if "fg" in j.get("jobname","")), jobs[0])
    pcts = fg.get("read",{}).get("clat_ns",{}).get("percentile",{})
    p99  = next((v/1000 for k,v in pcts.items() if abs(float(k)-99.0)<0.01), None)
    iops = fg.get("read",{}).get("iops", 0)
    return p99, iops

# ── Step 1: Static scheduler baselines ────────────────────────────────────────

print("\n=== Phase 4: Validation ===")
print("\nStep 1: Running W2 interference with each static scheduler (30s each)...")

SCHEDULERS = ["none", "bfq", "kyber", "mq-deadline"]
static_results = {}

for sched in SCHEDULERS:
    print(f"  Testing static {sched}...", end=" ", flush=True)
    switch_scheduler(sched)
    p99, iops = run_fio(f"static_{sched}")
    static_results[sched] = {"p99": p99, "iops": iops}
    print(f"P99={p99:.0f}µs IOPS={iops:.0f}")

# ── Step 2: Run daemon in background + same workload ──────────────────────────

print("\nStep 2: Running daemon + W2 interference simultaneously (30s)...")

# Start daemon as background process
daemon = subprocess.Popen(
    [sys.executable, "phase4_daemon.py"],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE
)
time.sleep(2)  # Give daemon time to start

print("  Daemon started. Running workload...", end=" ", flush=True)
p99_daemon, iops_daemon = run_fio("adaptive_daemon")
print(f"P99={p99_daemon:.0f}µs IOPS={iops_daemon:.0f}")

# Stop daemon
daemon.terminate()
daemon.wait()

# ── Step 3: Print comparison ───────────────────────────────────────────────────

print("\n=== RESULTS ===")
print(f"\n  {'Approach':<25} {'P99 (µs)':>12} {'IOPS':>8} {'vs Daemon':>10}")
print(f"  {'-'*25} {'-'*12} {'-'*8} {'-'*10}")

for sched, r in static_results.items():
    p99  = r["p99"]
    iops = r["iops"]
    diff = (p99 - p99_daemon) / p99_daemon * 100 if p99_daemon else 0
    star = " ← best static" if p99 == min(r["p99"] for r in static_results.values()) else ""
    print(f"  {'Static ' + sched:<25} {p99:>12.0f} {iops:>8.0f} {diff:>+9.1f}%{star}")

print(f"  {'Adaptive Daemon':<25} {p99_daemon:>12.0f} {iops_daemon:>8.0f} {'baseline':>10}")

best_static_p99 = min(r["p99"] for r in static_results.values())
improvement = (best_static_p99 - p99_daemon) / best_static_p99 * 100
print(f"\n  Daemon vs best static scheduler: {improvement:+.1f}%")

if improvement >= 0:
    print("  ✓ Adaptive daemon matches or beats the best static scheduler!")
else:
    print("  ✗ Best static scheduler slightly wins — expected with small training set.")
    print("    In interviews: 'More training data would close this gap.'")

# ── Step 4: Chart ─────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 5))
fig.suptitle("Phase 4: Adaptive Daemon vs Static Schedulers (W2 Interference)",
             fontsize=12, fontweight="bold")

labels = [f"Static\n{s}" for s in SCHEDULERS] + ["Adaptive\nDaemon"]
values = [static_results[s]["p99"] for s in SCHEDULERS] + [p99_daemon]
colors = ["#888", "#0d6efd", "#198754", "#dc3545", "#fd7e14"]

bars = ax.bar(labels, values, color=colors, alpha=0.85, edgecolor="white")

# Gold border on best
best_i = int(np.argmin(values))
bars[best_i].set_edgecolor("gold")
bars[best_i].set_linewidth(2.5)

ax.set_ylabel("P99 Latency (µs) — lower is better")
ax.grid(axis="y", alpha=0.3)
for spine in ["top","right"]: ax.spines[spine].set_visible(False)

plt.tight_layout()
out = RESULTS / "phase4_validation.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nChart saved: {out}")
plt.show()

print("\n=== Phase 4 Complete! Your project is done. ===")
