#!/usr/bin/env python3
"""Generate loss comparison plots for debug/loss_monitor subset logs."""

import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOG_DIR = "/work/PaddleX_QA_test_new_science_400-500"
MODELS = [
    "annular_ring-annular_ring-annular_ring",
    "ldc-ldc_2d",
]
THRESHOLD = 1e-2


def extract_loss(log_file):
    with open(log_file, errors="ignore") as f:
        content = f.read()
    losses = {}
    pattern = re.compile(
        r"\[step:\s+(\d+)\].*?loss:\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    )
    for match in pattern.finditer(content):
        losses[int(match.group(1))] = float(match.group(2))
    return losses


def plot_one(model):
    dy_file = os.path.join(LOG_DIR, f"{model}_debug_dy.log")
    cinn_file = os.path.join(LOG_DIR, f"{model}_debug_CINN.log")
    out_png = os.path.join(LOG_DIR, f"{model}_debug_loss_400-500.png")
    dy = extract_loss(dy_file)
    cinn = extract_loss(cinn_file)
    steps = sorted(s for s in set(dy) & set(cinn) if 400 <= s <= 500)
    dy_loss = np.array([dy[s] for s in steps])
    cinn_loss = np.array([cinn[s] for s in steps])
    delta = cinn_loss - dy_loss
    outliers = np.abs(delta) > THRESHOLD

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax1.plot(steps, dy_loss, "b-", label="dy", linewidth=1)
    ax1.plot(steps, cinn_loss, "r--", label="CINN", linewidth=1)
    ax1.set_title(f"{model} debug/loss_monitor Loss (Step 400-500)")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps, delta, "g-", linewidth=1)
    ax2.axhline(0, color="k", linewidth=0.5)
    ax2.axhline(THRESHOLD, color="r", linestyle=":", label=f"+{THRESHOLD:g}")
    ax2.axhline(-THRESHOLD, color="r", linestyle=":", label=f"-{THRESHOLD:g}")
    if outliers.any():
        ax2.scatter(np.array(steps)[outliers], delta[outliers], c="red", s=20)
    ax2.set_title(f"Delta Loss | outliers(>{THRESHOLD:g}): {int(outliers.sum())}/{len(steps)}")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("CINN - dy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=100)
    plt.close()

    return {
        "model": model,
        "common_steps": len(steps),
        "outliers": int(outliers.sum()),
        "max_abs_delta": float(np.max(np.abs(delta))) if len(delta) else None,
        "mean_abs_delta": float(np.mean(np.abs(delta))) if len(delta) else None,
        "plot": out_png,
    }


if __name__ == "__main__":
    results = [plot_one(model) for model in MODELS]
    for item in results:
        print(
            f"{item['model']}: common_steps={item['common_steps']} "
            f"outliers={item['outliers']} max_abs_delta={item['max_abs_delta']:.6g} "
            f"mean_abs_delta={item['mean_abs_delta']:.6g} plot={os.path.basename(item['plot'])}",
            flush=True,
        )
