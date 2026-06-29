#!/usr/bin/env python3
"""Rerun representative science models with debug=1 and loss_monitor=1.

This script intentionally uses unique hydra.run.dir per run/mode to avoid checkpoint
pollution between dy and CINN.
"""

import os
import re
import subprocess
import time

MODULUS_DIR = "/work/modulus-sym"
LOG_DIR = "/work/PaddleX_QA_test_new_science_400-500"
RUN_TAG = time.strftime("%Y%m%d_%H%M%S")

MODELS = [
    "annular_ring/annular_ring/annular_ring",
    "ldc/ldc_2d",
]

DY_ENV = {
    "FLAGS_set_to_1d": "0",
    "CUDA_MODULE_LOADING": "LAZY",
    "ENABLE_CINN_IN_DY2ST": "0",
    "DDE_BACKEND": "paddle",
    "NVIDIA_TF32_OVERRIDE": "1",
    "PROFILING_TIMER_ONLY": "no",
    "debug": "1",
    "loss_monitor": "1",
}

CINN_ENV = {
    "ENABLE_CINN_IN_DY2ST": "1",
    "FLAGS_enable_auto_recompute": "1",
    "FLAGS_prim_vjp_skip_default_ops": "False",
    "FLAGS_cinn_debug": "1",
    "FLAGS_enable_pir_in_executor": "true",
    "FLAGS_enable_pir_api": "True",
    "FLAGS_cinn_bucket_compile": "True",
    "FLAGS_group_schedule_tiling_first": "1",
    "FLAGS_cinn_new_group_scheduler": "1",
    "FLAGS_nvrtc_compile_to_cubin": "True",
    "DDE_BACKEND": "paddle",
    "NVIDIA_TF32_OVERRIDE": "1",
    "PROFILING_TIMER_ONLY": "no",
    "debug": "1",
    "loss_monitor": "1",
}


def get_train_cmd(dir_name, mode):
    tipc_root = "dynamicTostatic" if mode == "cinn" else "dynamic"
    rb = os.path.join(
        MODULUS_DIR,
        "test_tipc",
        tipc_root,
        dir_name,
        "benchmark_common/run_benchmark.sh",
    )
    content = open(rb).read()
    match = re.search(r'train_cmd="([^"]+)"', content)
    return match.group(1) if match else None


def build_train_cmd(model_path, mode):
    dir_name = model_path.replace("/", "-")
    train_cmd = get_train_cmd(dir_name, mode)
    if not train_cmd:
        raise RuntimeError(f"no train_cmd found for {dir_name} mode={mode}")

    train_cmd = re.sub(r"training\.max_steps=\d+", "training.max_steps=500", train_cmd)
    train_cmd = train_cmd.rstrip("; popd")
    hydra_dir = f"outputs_science_debug_loss_monitor/{RUN_TAG}/{dir_name}/{mode}"
    train_cmd += (
        " training.print_stats_freq=1"
        " training.rec_validation_freq=9999"
        " training.rec_inference_freq=9999"
        " training.rec_monitor_freq=9999"
        " training.save_network_freq=9999"
        f" hydra.run.dir={hydra_dir}"
        "; popd"
    )
    return train_cmd


def run_one(model_path, mode, env_vars):
    dir_name = model_path.replace("/", "-")
    suffix = "debug_dy" if mode == "dy" else "debug_CINN"
    log_file = os.path.join(LOG_DIR, f"{dir_name}_{suffix}.log")
    train_cmd = build_train_cmd(model_path, mode)

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/usr/lib64/:/usr/local/lib/:" + env.get("LD_LIBRARY_PATH", "")
    env["PATH"] = "/work/env3.10/bin:" + env.get("PATH", "")
    env.update(env_vars)
    env["to_static"] = "True" if mode == "cinn" else "False"

    with open(log_file, "w") as f:
        subprocess.run(
            ["bash", "-c", train_cmd],
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=MODULUS_DIR,
            env=env,
            timeout=1800,
            check=False,
        )
    return log_file


def check_log(log_file):
    content = open(log_file, errors="ignore").read()
    steps = re.findall(r"\[step:\s*(\d+)\]", content)
    losses = re.findall(r"\[step:\s*(\d+)\].*?loss:\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)", content)
    max_step = max((int(s) for s in steps), default=0)
    step0_loss = None
    for step, loss in losses:
        if int(step) == 0:
            step0_loss = float(loss)
            break
    has_to_static = "Using jit.to_static" in content
    restore_lines = [line for line in content.splitlines() if "attempting to restore from" in line][:3]
    return max_step, step0_loss, has_to_static, restore_lines


if __name__ == "__main__":
    print(f"RUN_TAG={RUN_TAG}", flush=True)
    results = []
    for model in MODELS:
        print(f"\n=== {model} ===", flush=True)
        for mode, env_vars in (("dy", DY_ENV), ("cinn", CINN_ENV)):
            print(f"  Running {mode}...", flush=True)
            try:
                log_file = run_one(model, mode, env_vars)
                max_step, step0_loss, has_to_static, restore_lines = check_log(log_file)
                print(
                    f"  {mode}: max_step={max_step}, step0_loss={step0_loss}, "
                    f"to_static_msg={has_to_static}, log={os.path.basename(log_file)}",
                    flush=True,
                )
                for line in restore_lines:
                    print(f"    {line}", flush=True)
                results.append((model, mode, max_step, step0_loss, has_to_static, log_file))
            except subprocess.TimeoutExpired:
                print(f"  {mode}: TIMEOUT", flush=True)
                results.append((model, mode, 0, None, False, "TIMEOUT"))
            except Exception as exc:
                print(f"  {mode}: ERROR {exc}", flush=True)
                results.append((model, mode, 0, None, False, f"ERROR {exc}"))

    print("\nSUMMARY")
    for item in results:
        print(item)
