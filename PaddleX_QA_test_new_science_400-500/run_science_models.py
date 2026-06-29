#!/usr/bin/env python3
"""Run 41 modulus-sym science models (dy + CINN) for 500 steps, save logs."""
import os, re, subprocess, sys

MODULUS_DIR = '/work/modulus-sym'
LOG_DIR = '/work/PaddleX_QA_test_new_science_400-500'
os.makedirs(LOG_DIR, exist_ok=True)

# 41 models: (plan_name, test_tipc_dir_name)
MODELS = [
    "ldc/ldc_2d", "chip_2d/chip_2d_solid_solid_heat_transfer",
    "turbulent_channel/2d_std_wf/u_tau_lookup", "turbulent_channel/2d/re590_k_om_LS",
    "three_fin_2d/heat_sink", "three_fin_3d/three_fin_flow",
    "chip_2d/chip_2d", "three_fin_2d/heat_sink_inverse",
    "chip_2d/chip_2d_solid_fluid_heat_transfer_heat", "turbulent_channel/2d/re590_k_ep_LS",
    "turbulent_channel/2d_std_wf/re590_k_ep", "turbulent_channel/2d_std_wf/re590_k_om",
    "three_fin_3d/three_fin_thermal", "ode_spring_mass/spring_mass_solver",
    "wave_equation/wave_1d", "ldc/ldc_2d_importance_sampling",
    "helmholtz/helmholtz", "ldc/ldc_2d_domain_decomposition",
    "surface_pde/sphere/sphere", "ldc/ldc_2d_domain_decomposition_fbpinn",
    "seismic_wave/wave_2d", "wave_equation/wave_1d_causal",
    "chip_2d/chip_2d_solid_fluid_heat_transfer_flow", "wave_equation/wave_inverse",
    "limerock/limerock_hFTB/limerock_thermal",
    "annular_ring/annular_ring_parameterized/annular_ring_parameterized",
    "annular_ring/annular_ring_equation_instancing/annular_ring",
    "annular_ring/annular_ring/annular_ring", "cylinder/cylinder_2d",
    "taylor_green/taylor_green_causal", "taylor_green/taylor_green",
    "annular_ring/annular_ring_gradient_enhanced/annular_ring_gradient_enhanced",
    "waveguide/slab_2D/slab_2D", "waveguide/cavity_2D/waveguide2D_TMz",
    "waveguide/cavity_3D/waveguide3D", "waveguide/slab_3D/slab_3D",
    "ldc/ldc_2d_zeroEq", "bracket/bracket", "fuselage_panel/panel",
    "anti_derivative/data_informed", "anti_derivative/physics_informed",
]

# Environment for dynamic graph
DY_ENV = {
    'FLAGS_set_to_1d': '0',
    'CUDA_MODULE_LOADING': 'LAZY',
    'ENABLE_CINN_IN_DY2ST': '0',
    'DDE_BACKEND': 'paddle',
    'NVIDIA_TF32_OVERRIDE': '1',
    'PROFILING_TIMER_ONLY': 'no',
    # Ensure deterministic / monitored loss behavior in modulus-sym science models.
    'debug': '1',
    'loss_monitor': '1',
}

# Environment for CINN (dy2st)
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
    # Ensure deterministic / monitored loss behavior in modulus-sym science models.
    'debug': '1',
    'loss_monitor': '1',
}

def get_train_cmd(dir_name, mode):
    """Extract python train command from run_benchmark.sh.

    dy uses test_tipc/dynamic, CINN uses test_tipc/dynamicTostatic to match the PLAN.
    """
    tipc_root = 'dynamicTostatic' if mode == 'cinn' else 'dynamic'
    rb = os.path.join(MODULUS_DIR, 'test_tipc', tipc_root, dir_name, 'benchmark_common/run_benchmark.sh')
    content = open(rb).read()
    m = re.search(r'train_cmd="([^"]+)"', content)
    if m:
        return m.group(1)
    return None

def run_model(model_path, mode, env_vars):
    """Run a model and return log file path."""
    dir_name = model_path.replace('/', '-')
    suffix = '_dy' if mode == 'dy' else '_CINN'
    log_file = os.path.join(LOG_DIR, f'{dir_name}{suffix}.log')

    train_cmd = get_train_cmd(dir_name, mode)
    if not train_cmd:
        print(f"  ERROR: no train_cmd found for {dir_name} mode={mode}")
        return None

    # Replace max_steps with 500 and add per-step logging.
    # Use isolated Hydra output dirs to avoid dy/CINN checkpoint pollution.
    train_cmd = re.sub(r'training\.max_steps=\d+', 'training.max_steps=500', train_cmd)
    train_cmd = train_cmd.rstrip('; popd')
    hydra_dir = f'outputs_science_400_500_debug_loss_monitor/{dir_name}/{mode}'
    train_cmd += (
        ' training.print_stats_freq=1'
        ' training.rec_validation_freq=9999'
        ' training.rec_inference_freq=9999'
        ' training.rec_monitor_freq=9999'
        ' training.save_network_freq=9999'
        f' hydra.run.dir={hydra_dir}'
        '; popd'
    )

    # Build environment
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = '/usr/lib64/:/usr/local/lib/:' + env.get('LD_LIBRARY_PATH', '')
    env['PATH'] = '/work/env3.10/bin:' + env.get('PATH', '')
    env.update(env_vars)

    # For CINN mode, set to_static=True
    if mode == 'cinn':
        env['to_static'] = 'True'
    else:
        env['to_static'] = 'False'

    with open(log_file, 'w') as f:
        proc = subprocess.run(
            ['bash', '-c', train_cmd],
            stdout=f, stderr=subprocess.STDOUT,
            cwd=MODULUS_DIR, env=env, timeout=1800
        )

    return log_file

def check_log(log_file):
    """Check if log has step 500 loss data."""
    if not log_file or not os.path.exists(log_file):
        return 0
    content = open(log_file, errors='ignore').read()
    steps = re.findall(r'\[step:\s*(\d+)\]', content)
    if steps:
        return max(int(s) for s in steps)
    return 0

if __name__ == '__main__':
    results = []
    total = len(MODELS)

    for idx, model in enumerate(MODELS):
        dir_name = model.replace('/', '-')
        print(f'\n=== [{idx+1}/{total}] {model} ===', flush=True)

        # Dynamic graph
        print(f'  Running dy...', flush=True)
        try:
            dy_log = run_model(model, 'dy', DY_ENV)
            dy_max = check_log(dy_log)
            print(f'  dy max_step={dy_max}', flush=True)
        except subprocess.TimeoutExpired:
            dy_max = 0
            print(f'  dy TIMEOUT', flush=True)
        except Exception as e:
            dy_max = 0
            print(f'  dy ERROR: {e}', flush=True)

        # CINN
        print(f'  Running CINN...', flush=True)
        try:
            cinn_log = run_model(model, 'cinn', CINN_ENV)
            cinn_max = check_log(cinn_log)
            print(f'  CINN max_step={cinn_max}', flush=True)
        except subprocess.TimeoutExpired:
            cinn_max = 0
            print(f'  CINN TIMEOUT', flush=True)
        except Exception as e:
            cinn_max = 0
            print(f'  CINN ERROR: {e}', flush=True)

        status = 'PASS' if dy_max >= 500 and cinn_max >= 500 else 'FAIL'
        results.append((model, dy_max, cinn_max, status))
        print(f'  => {status} (dy={dy_max}, cinn={cinn_max})', flush=True)

    # Summary
    print(f'\n\n{"="*60}')
    print(f'SUMMARY: {sum(1 for _,_,_,s in results if s=="PASS")}/{total} PASS')
    for m, d, c, s in results:
        print(f'  [{s}] {m}: dy={d} cinn={c}')
