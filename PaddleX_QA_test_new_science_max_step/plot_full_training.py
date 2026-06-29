#!/usr/bin/env python3
import os
import re
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LOG_DIR = '/work/PaddleX_QA_test_new_science_max_step'

MODELS = [
    ('ldc-ldc_2d', 'ldc/ldc_2d'),
    ('annular_ring-annular_ring-annular_ring', 'annular_ring/annular_ring'),
]

# Downsample: keep every N-th step for plotting
PLOT_STRIDE = 50

def extract_loss(path):
    points = []
    pat = re.compile(r'\[step:\s*(\d+)\].*?loss:\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)')
    with open(path, errors='ignore') as f:
        for line in f:
            m = pat.search(line)
            if m:
                try:
                    points.append((int(m.group(1)), float(m.group(2))))
                except ValueError:
                    pass
    # 去重并保持 step 有序；若同 step 多次出现，保留最后一次
    d = {}
    for s, v in points:
        d[s] = v
    return [(s, d[s]) for s in sorted(d)]

def plot_model(base, display_name):
    dy_log = os.path.join(LOG_DIR, base + '_dy.log')
    cinn_log = os.path.join(LOG_DIR, base + '_cinn.log')
    if not os.path.exists(dy_log) or not os.path.exists(cinn_log):
        print(f'SKIP {display_name}: missing log')
        return None
    dy = extract_loss(dy_log)
    cinn = extract_loss(cinn_log)
    dy_d = dict(dy)
    cinn_d = dict(cinn)
    common = sorted(set(dy_d) & set(cinn_d))
    if not common:
        print(f'SKIP {display_name}: no common steps')
        return None
    steps = np.array(common)
    dy_loss = np.array([dy_d[s] for s in common])
    cinn_loss = np.array([cinn_d[s] for s in common])
    delta = cinn_loss - dy_loss
    outliers = np.abs(delta) > 1e-2

    # Downsample for plotting clarity
    stride = PLOT_STRIDE
    ps = steps[::stride]
    pd = dy_loss[::stride]
    pc = cinn_loss[::stride]
    pdelta = delta[::stride]
    pout = outliers[::stride]

    suffix = 'full' if max(common) >= max(max(dy_d), max(cinn_d)) else f'partial_1-{max(common)}'
    out_png = os.path.join(LOG_DIR, base + f'_{suffix}_stride{stride}_loss_compare.png')

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    axes[0].plot(ps, pd, label='dy', linewidth=0.8)
    axes[0].plot(ps, pc, label='CINN', linewidth=0.8, linestyle='--')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{display_name} loss comparison ({suffix}, every {stride} steps, {len(ps)} pts)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].semilogy(ps, np.maximum(np.abs(pd), 1e-30), label='|dy loss|', linewidth=0.8)
    axes[1].semilogy(ps, np.maximum(np.abs(pc), 1e-30), label='|CINN loss|', linewidth=0.8, linestyle='--')
    axes[1].set_ylabel('|Loss| (log scale)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(ps, pdelta, label='CINN - dy', linewidth=0.8)
    axes[2].axhline(0, color='black', linewidth=0.5)
    axes[2].axhline(1e-2, color='red', linestyle=':', linewidth=0.8)
    axes[2].axhline(-1e-2, color='red', linestyle=':', linewidth=0.8)
    if pout.any():
        axes[2].scatter(ps[pout], pdelta[pout], color='red', s=8, zorder=3, label='|delta|>1e-2')
    axes[2].set_xlabel('Step')
    axes[2].set_ylabel('Delta loss')
    axes[2].set_title(f'Delta loss (sampled every {stride} steps), outliers(total): {int(outliers.sum())}/{len(delta)}')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    plt.close()

    print(f'OK {display_name}: common={len(common)}, stride={stride}, pts={len(ps)}, outliers(total)={int(outliers.sum())}, png={out_png}')
    return out_png

if __name__ == '__main__':
    for base, name in MODELS:
        plot_model(base, name)
