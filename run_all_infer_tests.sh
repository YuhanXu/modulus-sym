#!/bin/bash
# 科学计算模型推理性能+精度全量测试脚本
# 使用方法: bash run_all_infer_tests.sh

cd /work/modulus-sym

# 通用环境变量
export CUDA_VISIBLE_DEVICES=0
export PROFILING_TIMER_ONLY=no
export INFER_CHECK_ACCURACY=1
export INFER_ACCURACY_ITERS=50
export INFER_WARMUP=10
export INFER_STEPS=100

MODELS=(
    "annular_ring-annular_ring-annular_ring"
    "anti_derivative-data_informed"
    "bracket-bracket"
    "chip_2d-chip_2d"
    "cylinder-cylinder_2d"
    "fuselage_panel-panel"
    "helmholtz-helmholtz"
    "ldc-ldc_2d"
    "ode_spring_mass-spring_mass_solver"
    "seismic_wave-wave_2d"
    "surface_pde-sphere-sphere"
    "taylor_green-taylor_green"
    "three_fin_2d-heat_sink"
    "wave_equation-wave_1d"
    "wave_equation-wave_1d_causal"
    "wave_equation-wave_inverse"
    "waveguide-cavity_2D-waveguide2D_TMz"
    "waveguide-slab_2D-slab_2D"
)

echo "========== 动态图推理测试 =========="
export FLAGS_set_to_1d=0
export CUDA_MODULE_LOADING=LAZY
export ENABLE_CINN_IN_DY2ST=0

for model in "${MODELS[@]}"; do
    echo ""
    echo ">>> [动态图] ${model}"
    bash test_tipc/dynamic/${model}/N1C1/${model}_bs1_fp32_DP_infer.sh 2>&1 | tail -5
    echo ""
done

echo "========== 动转静+CINN 推理测试 =========="
export ENABLE_CINN_IN_DY2ST=1
export FLAGS_prim_vjp_skip_default_ops=False
export FLAGS_cinn_debug=1

for model in "${MODELS[@]}"; do
    echo ""
    echo ">>> [动转静+CINN] ${model}"
    bash test_tipc/dynamicTostatic/${model}/N1C1/${model}_bs1_fp32_DP_infer.sh 2>&1 | tail -5
    echo ""
done

echo "========== 全部测试完成 =========="
echo "日志文件位于 /work/modulus-sym/ 下："
echo "  动态图: modulus-sym_*_N1C1_infer_log"
echo "  动转静: modulus-sym_*_N1C1_d2sT_infer_log"
