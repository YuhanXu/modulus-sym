#!/usr/bin/env python3
"""Generate loss comparison plots (step 400-500) for modulus-sym science models.

Input logs are expected in LOG_DIR with names:
  <model_name>_dy.log
  <model_name>_CINN.log

Output plots and summary are written to LOG_DIR.
"""

import glob
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOG_DIR = "/work/PaddleX_QA_test_new_science_400-500"
SUMMARY_FILE = os.path.join(LOG_DIR, "science_loss_400-500_summary.txt")
THRESHOLD = 1e-2
PASS_OUTLIER_LIMIT = 10


def extract_loss(log_file):
    """Extract (step, loss) pairs from modulus-sym logs."""
    with open(log_file, errors="ignore") as f:
        content = f.read()

    results = []
    # Typical format:
    # [step:        400] lr: 0.0010000000, loss: 0.1234567890
    pattern = re.compile(
        r"\[step:\s+(\d+)\].*?loss:\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    )
    for match in pattern.finditer(content):
        try:
            results.append((int(match.group(1)), float(match.group(2))))
        except ValueError:
            pass

    # Keep the last occurrence per step if a log line is duplicated.
    dedup = {}
    for step, loss in results:
        dedup[step] = loss
    return sorted(dedup.items())


def plot_model(model_name, dy_file, cinn_file, output_png):
    dy_data = extract_loss(dy_file)
    cinn_data = extract_loss(cinn_file)
    dy_400 = [(step, loss) for step, loss in dy_data if 400 <= step <= 500]
    cinn_400 = [(step, loss) for step, loss in cinn_data if 400 <= step <= 500]

    if not dy_400 or not cinn_400:
        return {
            "model": model_name,
            "status": "SKIP",
            "reason": f"missing 400-500 loss points: dy={len(dy_400)} cinn={len(cinn_400)}",
            "dy_points": len(dy_400),
            "cinn_points": len(cinn_400),
            "common_steps": 0,
            "outliers": None,
        }

    dy_dict = dict(dy_400)
    cinn_dict = dict(cinn_400)
    common_steps = sorted(set(dy_dict) & set(cinn_dict))
    if len(common_steps) < 5:
        return {
            "model": model_name,
            "status": "SKIP",
            "reason": f"too few common steps: {len(common_steps)}",
            "dy_points": len(dy_400),
            "cinn_points": len(cinn_400),
            "common_steps": len(common_steps),
            "outliers": None,
        }

    steps = np.array(common_steps)
    dy_loss = np.array([dy_dict[step] for step in common_steps])
    cinn_loss = np.array([cinn_dict[step] for step in common_steps])
    delta = cinn_loss - dy_loss
    outliers = np.abs(delta) > THRESHOLD
    outlier_count = int(outliers.sum())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax1.plot(steps, dy_loss, "b-", label="Dynamic Graph", linewidth=1)
    ax1.plot(steps, cinn_loss, "r--", label="CINN", linewidth=1)
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{model_name} Loss Comparison (Step 400-500)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps, delta, "g-", linewidth=1)
    ax2.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
    ax2.axhline(y=THRESHOLD, color="r", linestyle=":", label=f"+{THRESHOLD:g}")
    ax2.axhline(y=-THRESHOLD, color="r", linestyle=":", label=f"-{THRESHOLD:g}")
    if outliers.any():
        ax2.scatter(steps[outliers], delta[outliers], c="red", zorder=5, s=20)
        for step, diff in zip(steps[outliers], delta[outliers]):
            ax2.annotate(
                f"({step},{diff:.4g})",
                (step, diff),
                fontsize=6,
                ha="center",
                va="bottom",
            )
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Delta Loss (CINN - dy)")
    ax2.set_title(f"Delta Loss | outliers(>{THRESHOLD:g}): {outlier_count}/{len(delta)}")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_png, dpi=100)
    plt.close()

    return {
        "model": model_name,
        "status": "PASS" if outlier_count <= PASS_OUTLIER_LIMIT else "FAIL",
        "reason": "",
        "dy_points": len(dy_400),
        "cinn_points": len(cinn_400),
        "common_steps": len(common_steps),
        "outliers": outlier_count,
        "plot": output_png,
    }


def main():
    dy_logs = sorted(glob.glob(os.path.join(LOG_DIR, "*_dy.log")))
    results = []

    for dy_file in dy_logs:
        model_name = os.path.basename(dy_file).replace("_dy.log", "")
        cinn_file = os.path.join(LOG_DIR, f"{model_name}_CINN.log")
        if not os.path.exists(cinn_file):
            results.append(
                {
                    "model": model_name,
                    "status": "SKIP",
                    "reason": "missing CINN log",
                    "dy_points": 0,
                    "cinn_points": 0,
                    "common_steps": 0,
                    "outliers": None,
                }
            )
            continue

        output_png = os.path.join(LOG_DIR, f"{model_name}_loss_400-500.png")
        result = plot_model(model_name, dy_file, cinn_file, output_png)
        results.append(result)
        if result["status"] == "SKIP":
            print(f"  SKIP {model_name}: {result['reason']}", flush=True)
        else:
            print(
                f"  {result['status']} {model_name}: common_steps={result['common_steps']} "
                f"outliers={result['outliers']} plot={os.path.basename(output_png)}",
                flush=True,
            )

    pass_count = sum(1 for item in results if item["status"] == "PASS")
    fail_count = sum(1 for item in results if item["status"] == "FAIL")
    skip_count = sum(1 for item in results if item["status"] == "SKIP")
    generated_count = pass_count + fail_count

    lines = []
    lines.append("Science model loss comparison summary (step 400-500)")
    lines.append(f"LOG_DIR={LOG_DIR}")
    lines.append(f"threshold={THRESHOLD:g}, pass_outlier_limit={PASS_OUTLIER_LIMIT}")
    lines.append(f"Generated plots: {generated_count}")
    lines.append(f"PASS={pass_count} FAIL={fail_count} SKIP={skip_count} Total logs={len(results)}")
    lines.append("")
    for item in sorted(results, key=lambda x: (x["status"], x["model"])):
        if item["status"] == "SKIP":
            lines.append(f"[{item['status']}] {item['model']}: {item['reason']}")
        else:
            lines.append(
                f"[{item['status']}] {item['model']}: common_steps={item['common_steps']} "
                f"outliers={item['outliers']} dy_points={item['dy_points']} cinn_points={item['cinn_points']}"
            )

    summary = "\n".join(lines) + "\n"
    with open(SUMMARY_FILE, "w") as f:
        f.write(summary)

    print("\n" + summary, flush=True)
    print(f"Summary written to {SUMMARY_FILE}", flush=True)


if __name__ == "__main__":
    main()
