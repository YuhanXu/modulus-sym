#!/bin/bash
# 新增23个模型推理性能测试
cd /work/modulus-sym

export LD_LIBRARY_PATH=/usr/lib64:${LD_LIBRARY_PATH}
export CUDA_VISIBLE_DEVICES=0
export PROFILING_TIMER_ONLY=no
export INFER_WARMUP=10
export INFER_STEPS=100
export debug=1
export loss_monitor=1

NEW23=(
    "ldc-ldc_2d_importance_sampling"
    "ldc-ldc_2d_domain_decomposition"
    "ldc-ldc_2d_domain_decomposition_fbpinn"
    "ldc-ldc_2d_zeroEq"
    "chip_2d-chip_2d_solid_solid_heat_transfer"
    "chip_2d-chip_2d_solid_fluid_heat_transfer_heat"
    "chip_2d-chip_2d_solid_fluid_heat_transfer_flow"
    "annular_ring-annular_ring_parameterized-annular_ring_parameterized"
    "annular_ring-annular_ring_equation_instancing-annular_ring"
    "annular_ring-annular_ring_gradient_enhanced-annular_ring_gradient_enhanced"
    "limerock-limerock_hFTB-limerock_thermal"
    "taylor_green-taylor_green_causal"
    "three_fin_2d-heat_sink_inverse"
    "three_fin_3d-three_fin_flow"
    "three_fin_3d-three_fin_thermal"
    "waveguide-cavity_3D-waveguide3D"
    "waveguide-slab_3D-slab_3D"
    "turbulent_channel-2d_std_wf-u_tau_lookup"
    "turbulent_channel-2d-re590_k_om_LS"
    "turbulent_channel-2d-re590_k_ep_LS"
    "turbulent_channel-2d_std_wf-re590_k_ep"
    "turbulent_channel-2d_std_wf-re590_k_om"
    "anti_derivative-physics_informed"
)

echo "========== 动态图推理测试 =========="
export FLAGS_set_to_1d=0
export CUDA_MODULE_LOADING=LAZY
export ENABLE_CINN_IN_DY2ST=0

for model in "${NEW23[@]}"; do
    echo ""
    echo ">>> [动态图] ${model}"
    bash test_tipc/dynamic/${model}/N1C1/${model}_bs1_fp32_DP_infer.sh 2>&1 | tail -5
    echo ""
done

echo "========== 动转静+CINN 推理测试 =========="
export ENABLE_CINN_IN_DY2ST=1
export FLAGS_prim_vjp_skip_default_ops=False
export FLAGS_cinn_debug=1

for model in "${NEW23[@]}"; do
    echo ""
    echo ">>> [动转静+CINN] ${model}"
    bash test_tipc/dynamicTostatic/${model}/N1C1/${model}_bs1_fp32_DP_infer.sh 2>&1 | tail -5
    echo ""
done

echo "========== 完成 =========="
