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
export debug=1
export loss_monitor=1

MODELS=(
    "ldc-ldc_2d"
    "chip_2d-chip_2d_solid_solid_heat_transfer"
    "turbulent_channel-2d_std_wf-u_tau_lookup"
    "turbulent_channel-2d-re590_k_om_LS"
    "three_fin_2d-heat_sink"
    "three_fin_3d-three_fin_flow"
    "chip_2d-chip_2d"
    "three_fin_2d-heat_sink_inverse"
    "chip_2d-chip_2d_solid_fluid_heat_transfer_heat"
    "turbulent_channel-2d-re590_k_ep_LS"
    "turbulent_channel-2d_std_wf-re590_k_ep"
    "turbulent_channel-2d_std_wf-re590_k_om"
    "three_fin_3d-three_fin_thermal"
    "ode_spring_mass-spring_mass_solver"
    "wave_equation-wave_1d"
    "ldc-ldc_2d_importance_sampling"
    "helmholtz-helmholtz"
    "ldc-ldc_2d_domain_decomposition"
    "surface_pde-sphere-sphere"
    "ldc-ldc_2d_domain_decomposition_fbpinn"
    "seismic_wave-wave_2d"
    "wave_equation-wave_1d_causal"
    "chip_2d-chip_2d_solid_fluid_heat_transfer_flow"
    "wave_equation-wave_inverse"
    "limerock-limerock_hFTB-limerock_thermal"
    "annular_ring-annular_ring_parameterized-annular_ring_parameterized"
    "annular_ring-annular_ring_equation_instancing-annular_ring"
    "annular_ring-annular_ring-annular_ring"
    "cylinder-cylinder_2d"
    "taylor_green-taylor_green_causal"
    "taylor_green-taylor_green"
    "annular_ring-annular_ring_gradient_enhanced-annular_ring_gradient_enhanced"
    "waveguide-slab_2D-slab_2D"
    "waveguide-cavity_2D-waveguide2D_TMz"
    "waveguide-cavity_3D-waveguide3D"
    "waveguide-slab_3D-slab_3D"
    "ldc-ldc_2d_zeroEq"
    "bracket-bracket"
    "fuselage_panel-panel"
    "anti_derivative-data_informed"
    "anti_derivative-physics_informed"
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
