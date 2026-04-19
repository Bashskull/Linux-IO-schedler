#!/usr/bin/env python3
"""
Phase 3: ML Model — Predict Optimal Scheduler Per Workload
===========================================================
Reads all benchmark results (Phase 1 + Phase 3 workloads),
builds features, trains a decision tree, validates it, prints results.

What the model learns:
  INPUT:  workload characteristics (read ratio, queue depth, block size, etc.)
  OUTPUT: which scheduler gives the lowest P99 latency

Run: python3 phase3_ml.py
"""

import json, glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

RESULTS_DIR  = Path.home() / "results"
PHASE3_DIR   = RESULTS_DIR / "phase3"
PHASE1_DIR   = RESULTS_DIR  # Phase 1 JSONs are directly in ~/results/

SCHEDULERS   = ["none", "bfq", "kyber", "mq-deadline"]
COLORS       = {"none": "#888", "bfq": "#0d6efd", "kyber": "#198754", "mq-deadline": "#dc3545"}

# ── Workload feature definitions ───────────────────────────────────────────────
# These are the features the ML model uses to make its decision.
# In a real system you'd extract these from blktrace — here we define them
# from what we know about each workload we designed.
#
# Features:
#   read_ratio    : fraction of I/O that is reads (0.0 to 1.0)
#   queue_depth   : avg number of in-flight I/O requests
#   block_size_kb : size of each I/O request in KB
#   num_jobs      : number of concurrent processes
#   has_interference: 1 if background I/O is competing, 0 if not

WORKLOAD_FEATURES = {
    "W1_latency":    {"read_ratio": 1.0, "queue_depth": 1,  "block_size_kb": 4,   "num_jobs": 1, "has_interference": 0},
    "W2_interference":{"read_ratio": 0.3, "queue_depth": 33, "block_size_kb": 4,   "num_jobs": 3, "has_interference": 1},
    "W3_sequential": {"read_ratio": 1.0, "queue_depth": 32, "block_size_kb": 128, "num_jobs": 1, "has_interference": 0},
    "W4_randwrite":  {"read_ratio": 0.0, "queue_depth": 64, "block_size_kb": 4,   "num_jobs": 2, "has_interference": 0},
    "W5_concurrent": {"read_ratio": 1.0, "queue_depth": 32, "block_size_kb": 4,   "num_jobs": 4, "has_interference": 0},
}

# ── Parse fio JSON ─────────────────────────────────────────────────────────────

def parse(filepath, is_interference=False):
    """Extract IOPS and P99 from a fio JSON file."""
    try:
        data = json.load(open(filepath))
    except:
        return None

    jobs = data.get("jobs", [])
    if not jobs:
        return None

    if is_interference:
        job = next((j for j in jobs if "fg" in j.get("jobname", "")), jobs[0])
    else:
        job = jobs[0]

    read  = job.get("read", {})
    write = job.get("write", {})

    # Use read P99 if reads exist, else write P99
    stats = read if read.get("iops", 0) > 0 else write
    iops  = stats.get("iops", 0)
    pcts  = stats.get("clat_ns", {}).get("percentile", {})
    p99   = next((v/1000 for k,v in pcts.items() if abs(float(k)-99.0)<0.01), None)

    return {"iops": iops, "p99_us": p99}

# ── Collect all results ────────────────────────────────────────────────────────

records = []

# Phase 1 workloads (W1, W2) — in ~/results/
for wkey in ["W1_latency", "W2_interference"]:
    is_interf = (wkey == "W2_interference")
    for sched in SCHEDULERS:
        path = PHASE1_DIR / f"{wkey}_{sched}.json"
        if not path.exists():
            print(f"Missing: {path.name}")
            continue
        m = parse(path, is_interference=is_interf)
        if m:
            records.append({
                "workload": wkey,
                "scheduler": sched,
                **WORKLOAD_FEATURES[wkey],
                **m
            })

# Phase 3 workloads (W3, W4, W5) — in ~/results/phase3/
for wkey in ["W3_sequential", "W4_randwrite", "W5_concurrent"]:
    for sched in SCHEDULERS:
        path = PHASE3_DIR / f"{wkey}_{sched}.json"
        if not path.exists():
            print(f"Missing: {path.name}")
            continue
        m = parse(path)
        if m:
            records.append({
                "workload": wkey,
                "scheduler": sched,
                **WORKLOAD_FEATURES[wkey],
                **m
            })

df = pd.DataFrame(records)
print(f"\nLoaded {len(df)} benchmark results across {df['workload'].nunique()} workloads\n")

# ── Build training labels ──────────────────────────────────────────────────────
# For each workload, label = the scheduler with the lowest P99

best_per_workload = df.loc[df.groupby("workload")["p99_us"].idxmin()][["workload","scheduler","p99_us"]]
best_per_workload.columns = ["workload", "best_scheduler", "best_p99_us"]

print("Training Labels (best scheduler per workload):")
print("-" * 55)
for _, row in best_per_workload.iterrows():
    print(f"  {row['workload']:<20} → {row['best_scheduler']:<14} (P99={row['best_p99_us']:.0f}µs)")
print()

# ── Build feature matrix ───────────────────────────────────────────────────────

# One row per workload (not per scheduler) — label is the best scheduler
feature_cols = ["read_ratio", "queue_depth", "block_size_kb", "num_jobs", "has_interference"]

# Get unique workload feature rows
workload_df = df[["workload"] + feature_cols].drop_duplicates("workload")
train_df    = workload_df.merge(best_per_workload, on="workload")

X = train_df[feature_cols].values
y = train_df["best_scheduler"].values

print("Feature matrix (what the model sees):")
print("-" * 70)
print(f"  {'Workload':<20} {'read%':>6} {'QD':>5} {'BS(KB)':>8} {'jobs':>6} {'interf':>7} → Label")
print(f"  {'-'*20} {'-'*6} {'-'*5} {'-'*8} {'-'*6} {'-'*7}   {'-'*14}")
for _, row in train_df.iterrows():
    print(f"  {row['workload']:<20} {row['read_ratio']:>5.0%} {row['queue_depth']:>5} "
          f"{row['block_size_kb']:>8} {row['num_jobs']:>6} {row['has_interference']:>7} → {row['best_scheduler']}")
print()

# ── Train model ────────────────────────────────────────────────────────────────
# Decision tree — interpretable, fast, perfect for this feature space

# With only 5 workloads we train on all and do leave-one-out validation
# (too few samples for a proper 70/30 split — we're honest about this)

clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X, y)

print("Decision Tree Rules (how the model thinks):")
print("-" * 55)
tree_rules = export_text(clf, feature_names=feature_cols)
print(tree_rules)

# ── Validation: leave-one-out ──────────────────────────────────────────────────

print("Validation (Leave-One-Out — each workload tested as unseen):")
print("-" * 65)
print(f"  {'Workload':<20} {'Actual':>14} {'Predicted':>14} {'Correct':>8}")
print(f"  {'-'*20} {'-'*14} {'-'*14} {'-'*8}")

correct = 0
for i in range(len(X)):
    # Train on everything except row i
    X_train = np.delete(X, i, axis=0)
    y_train = np.delete(y, i, axis=0)
    X_test  = X[i].reshape(1, -1)
    y_test  = y[i]

    clf_loo = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf_loo.fit(X_train, y_train)
    pred = clf_loo.predict(X_test)[0]

    ok = "✓" if pred == y_test else "✗"
    if pred == y_test:
        correct += 1
    print(f"  {train_df.iloc[i]['workload']:<20} {y_test:>14} {pred:>14} {ok:>8}")

accuracy = correct / len(X) * 100
print(f"\n  Leave-One-Out Accuracy: {correct}/{len(X)} = {accuracy:.0f}%")
print()

# ── P99 improvement: model vs always picking one static scheduler ──────────────

print("Model vs Static Scheduler (avg P99 across all workloads):")
print("-" * 55)

# For each workload, get P99 of model's predicted best vs each static choice
workload_p99 = df.pivot_table(index="workload", columns="scheduler", values="p99_us")

model_p99s = []
for _, row in train_df.iterrows():
    wk   = row["workload"]
    pred = clf.predict(row[feature_cols].values.reshape(1,-1))[0]
    p99  = workload_p99.loc[wk, pred] if pred in workload_p99.columns else np.nan
    model_p99s.append(p99)

model_avg = np.nanmean(model_p99s)

print(f"  {'Approach':<25} {'Avg P99 (µs)':>14} {'vs Model':>10}")
print(f"  {'-'*25} {'-'*14} {'-'*10}")
print(f"  {'ML Model (adaptive)':<25} {model_avg:>14.0f} {'—':>10}")

for sched in SCHEDULERS:
    if sched not in workload_p99.columns:
        continue
    avg = workload_p99[sched].mean()
    improvement = (avg - model_avg) / avg * 100
    print(f"  {'Static ' + sched:<25} {avg:>14.0f} {improvement:>+9.1f}%")

print()

# ── Save model visually ────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Phase 3: ML Model — Optimal Scheduler Prediction", fontsize=13, fontweight="bold")

# Left: Decision tree visualization
ax1 = axes[0]
plot_tree(clf, feature_names=feature_cols,
          class_names=clf.classes_,
          filled=True, rounded=True, ax=ax1, fontsize=9)
ax1.set_title("Decision Tree Structure", fontsize=11)

# Right: P99 comparison — model vs static schedulers per workload
ax2 = axes[1]
workloads = list(WORKLOAD_FEATURES.keys())
x = np.arange(len(workloads))
width = 0.18

for i, sched in enumerate(SCHEDULERS):
    vals = [workload_p99.loc[w, sched] if w in workload_p99.index and sched in workload_p99.columns else 0
            for w in workloads]
    ax2.bar(x + i*width, vals, width, label=sched, color=COLORS[sched], alpha=0.8)

# Model predictions
model_vals = []
for _, row in train_df.iterrows():
    pred = clf.predict(row[feature_cols].values.reshape(1,-1))[0]
    p99  = workload_p99.loc[row["workload"], pred] if pred in workload_p99.columns else 0
    model_vals.append(p99)

ax2.plot(x + 1.5*width, model_vals, "D--", color="black",
         markersize=8, linewidth=2, label="ML Model pick", zorder=5)

ax2.set_title("P99 per Workload: Static vs ML Model Pick", fontsize=10)
ax2.set_xticks(x + 1.5*width)
ax2.set_xticklabels([w.replace("_", "\n") for w in workloads], fontsize=8)
ax2.set_ylabel("P99 Latency (µs)")
ax2.legend(fontsize=8)
ax2.grid(axis="y", alpha=0.3)
for spine in ["top","right"]: ax2.spines[spine].set_visible(False)

plt.tight_layout()
out = RESULTS_DIR / "phase3_model.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Chart saved: {out}")
plt.show()

print("\n=== Phase 3 Complete! ===")
print("  You now have a trained decision tree that predicts optimal scheduler.")
print("  Ready for Phase 4: the daemon that switches schedulers automatically.")
