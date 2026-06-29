#!/usr/bin/env python3
"""Run limerock/limerock_hFTB/limerock_thermal full training in dy and CINN modes."""
import os
import re
import subprocess

MODULUS_DIR = '/work/modulus-sym'
LOG_DIR = '/work/PaddleX_QA_test_new_science_max_step'
MODEL_PATH = 'limerock/limerock_hFTB/limerock_thermal'
DIR_NAME = 'limerock-limerock_hFTB-limerock_thermal'

DY_ENV = {
    'FLAGS_set_to_1d': '0',
    'CUDA_MODULE_LOADING': 'LAZY',
    'ENABLE_CINN_IN_DY2ST': '0',
    'DDE_BACKEND': 'paddle',
    'NVIDIA_TF32_OVERRIDE': '1',
    'PROFILING_TIMER_ONLY': 'no',
    'debug': '1',
    'loss_monitor': '1',
}

CINN_ENV = {
    'ENABLE_CINN_IN_DY2ST': '1',
    'FLAGS_enable_auto_recompute': '1',
    'FLAGS_prim_vjp_skip_default_ops': 'False',
    'FLAGS_cinn_debug': '1',
    'FLAGS_enable_pir_in_executor': 'true',
    'FLAGS_enable_pir_api': 'True',
    'FLAGS_cinn_bucket_compile': 'True',
    'FLAGS_group_schedule_tiling_first': '1',
    'FLAGS_cinn_new_group_scheduler': '1',
    'FLAGS_nvrtc_compile_to_cubin': 'True',
    'DDE_BACKEND': 'paddle',
    'NVIDIA_TF32_OVERRIDE': '1',
    'PROFILING_TIMER_ONLY': 'no',
    'debug': '1',
    'loss_monitor': '1',
}


def build_env(env_vars, mode):
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = '/usr/lib64/:/usr/local/lib/:' + env.get('LD_LIBRARY_PATH', '')
    env['PATH'] = '/work/env3.10/bin:' + env.get('PATH', '')
    env.update(env_vars)
    env['to_static'] = 'True' if mode == 'cinn' else 'False'
    return env


def get_train_cmd(mode):
    tipc_root = 'dynamicTostatic' if mode == 'cinn' else 'dynamic'
    run_benchmark = os.path.join(
        MODULUS_DIR,
        'test_tipc',
        tipc_root,
        DIR_NAME,
        'benchmark_common/run_benchmark.sh',
    )
    with open(run_benchmark) as f:
        match = re.search(r'train_cmd="([^"]+)"', f.read())
    if not match:
        raise RuntimeError(f'Cannot find train_cmd in {run_benchmark}')
    train_cmd = match.group(1)
    train_cmd = re.sub(r'\s*training\.max_steps=\d+', '', train_cmd)
    train_cmd = train_cmd.rstrip('; popd')
    hydra_dir = f'outputs_science_max_step/{DIR_NAME}/{mode}'
    return (
        train_cmd
        + ' training.print_stats_freq=1'
        + ' training.rec_validation_freq=9999'
        + ' training.rec_inference_freq=9999'
        + ' training.rec_monitor_freq=9999'
        + ' training.save_network_freq=9999'
        + f' hydra.run.dir={hydra_dir}'
        + '; popd'
    )


def run(mode, env_vars):
    log_file = os.path.join(LOG_DIR, f'{DIR_NAME}_{mode}.log')
    train_cmd = get_train_cmd(mode)
    env = build_env(env_vars, mode)
    print(f'[{mode}] {MODEL_PATH} log={log_file}', flush=True)
    print(f'  cmd: {train_cmd}', flush=True)
    with open(log_file, 'w') as f:
        subprocess.run(
            ['bash', '-c', train_cmd],
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=MODULUS_DIR,
            env=env,
            check=False,
        )
    with open(log_file, errors='ignore') as f:
        steps = re.findall(r'\[step:\s*(\d+)\]', f.read())
    max_step = max(int(step) for step in steps) if steps else 0
    print(f'  => max_step={max_step}', flush=True)
    return max_step


if __name__ == '__main__':
    print(f'=== {MODEL_PATH} ===', flush=True)
    for mode, env_vars in [('dy', DY_ENV), ('cinn', CINN_ENV)]:
        try:
            run(mode, env_vars)
        except Exception as exc:
            print(f'  {mode} ERROR: {exc}', flush=True)
    print('\nAll done.', flush=True)
