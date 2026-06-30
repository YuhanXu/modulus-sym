# 科学计算模型 modulus-sym 推理性能测试方案

## 1. 环境准备（已完成）

modulus-sym 模型仓已下载，目录：`/work/modulus-sym`

```bash
cd /work/modulus-sym
export PROFILING_TIMER_ONLY=no
```

## 2. 推理脚本修改要点（已完成）

基于训练脚本进行推理改造，核心差异：

| 修改项 | 训练模式 | 推理模式 |
|--------|----------|----------|
| 梯度计算 | 开启 | `paddle.no_grad()` 关闭 |
| 模型状态 | `model.train()` | `model.eval()` |
| 反向传播 | `loss.backward()` | 删除 |
| 优化器步进 | `optimizer.step()` | 删除 |
| 性能指标 | 训练吞吐 (ms/iteration) | 推理延迟 (ms/iteration) + 吞吐 (iterations/s) |
| Warmup | 少量 | 充分预热（≥10 iter）以稳定 kernel cache |

### 已生成文件列表

| 文件 | 说明 |
|------|------|
| `examples/annular_ring/annular_ring/annular_ring_infer.py` | 推理 Python 脚本 |
| `test_tipc/dynamic/.../benchmark_common_infer/run_benchmark.sh` | 动态图推理 benchmark runner |
| `test_tipc/dynamic/.../N1C1/..._bs1_fp32_DP_infer.sh` | 动态图推理入口脚本 |
| `test_tipc/dynamicTostatic/.../benchmark_common_infer/run_benchmark.sh` | 动转静推理 benchmark runner |
| `test_tipc/dynamicTostatic/.../N1C1/..._bs1_fp32_DP_infer.sh` | 动转静推理入口脚本 |

## 3. 运行方式

### 3.1 动态图推理（算子库）— 性能测试

```bash
cd /work/modulus-sym
export CUDA_VISIBLE_DEVICES=0
export FLAGS_set_to_1d=0
export CUDA_MODULE_LOADING=LAZY
export ENABLE_CINN_IN_DY2ST=0
export PROFILING_TIMER_ONLY=no

bash test_tipc/dynamic/annular_ring-annular_ring-annular_ring/N1C1/annular_ring-annular_ring-annular_ring_bs1_fp32_DP_infer.sh
```

### 3.2 动转静 + CINN 编译器推理 — 性能测试

```bash
cd /work/modulus-sym
export CUDA_VISIBLE_DEVICES=0
export ENABLE_CINN_IN_DY2ST=1
export FLAGS_prim_vjp_skip_default_ops=False
export FLAGS_cinn_debug=1

bash test_tipc/dynamicTostatic/annular_ring-annular_ring-annular_ring/N1C1/annular_ring-annular_ring-annular_ring_bs1_fp32_DP_infer.sh
```

### 3.3 动态图推理（算子库）— 精度测试

```bash
cd /work/modulus-sym
export CUDA_VISIBLE_DEVICES=0
export FLAGS_set_to_1d=0
export CUDA_MODULE_LOADING=LAZY
export ENABLE_CINN_IN_DY2ST=0
export PROFILING_TIMER_ONLY=no
export INFER_CHECK_ACCURACY=1
export INFER_ACCURACY_ITERS=50

bash test_tipc/dynamic/annular_ring-annular_ring-annular_ring/N1C1/annular_ring-annular_ring-annular_ring_bs1_fp32_DP_infer.sh
```

### 3.4 动转静 + CINN 编译器推理 — 精度测试

```bash
cd /work/modulus-sym
export CUDA_VISIBLE_DEVICES=0
export ENABLE_CINN_IN_DY2ST=1
export FLAGS_prim_vjp_skip_default_ops=False
export FLAGS_cinn_debug=1
export INFER_CHECK_ACCURACY=1
export INFER_ACCURACY_ITERS=50

bash test_tipc/dynamicTostatic/annular_ring-annular_ring-annular_ring/N1C1/annular_ring-annular_ring-annular_ring_bs1_fp32_DP_infer.sh
```

### 3.5 环境变量说明

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `INFER_WARMUP` | 10 | warmup 迭代数 |
| `INFER_STEPS` | 100 | benchmark 迭代数 |
| `INFER_CHECK_ACCURACY` | 0 | 设为 1 开启精度对比（在性能计时之后执行，不影响性能数据） |
| `INFER_ACCURACY_ITERS` | 50 | 精度检查迭代数 |

## 4. 性能指标采集

| 指标 | 说明 |
|------|------|
| avg_latency (ms) | 单次前向推理平均耗时（`time/iteration:` 关键字） |
| throughput (iterations/s) | 每秒推理迭代数 |
| gpu_memory_peak (MB) | GPU 显存峰值 |

日志输出格式与训练一致（`time/iteration: xxx`），可复用现有 `analysis_log.py` 解析。

## 5. 对比维度

| 对比组 | 场景 |
|--------|------|
| 动态图 vs 动转静+CINN | 推理加速比 |
| 训练性能 vs 推理性能 | 去除反向后的加速效果 |

## 6. 单模型验证结果（annular_ring, A100-80GB）

| 指标 | 动态图（算子库） | 动转静 + CINN | 加速比 |
|------|-----------------|---------------|--------|
| avg_latency | 24.90 ms | 10.71 ms | **2.32x** |
| throughput | 40.15 iter/s | 93.38 iter/s | 2.32x |
| GPU 显存峰值 | 5086.8 MB | 2327.5 MB | 节省 54% |
| CINN 编译耗时 | - | ~90s（首次） | - |

关键发现：
- CINN 编译器推理延迟降低 57%，加速 2.32 倍
- CINN 模式显存占用更低（算子融合减少中间 tensor）
- 首次运行有 ~90s 编译开销，后续命中缓存

## 7. 全量测试结果（18模型，A100-80GB）

### 7.1 性能对比（time/iteration, ms）

| # | 模型 | 动态图(ms) | 动转静+CINN(ms) | 加速比 |
|---|------|-----------|----------------|--------|
| 1 | annular_ring | 28.89 | 10.70 | 2.70x |
| 2 | anti_derivative | 17.21 | 14.80 | 1.16x |
| 3 | bracket | 54.13 | 19.78 | 2.74x |
| 4 | chip_2d | 47.20 | 16.08 | 2.94x |
| 5 | cylinder_2d | 27.76 | 11.24 | 2.47x |
| 6 | fuselage_panel | 43.82 | 24.99 | 1.75x |
| 7 | helmholtz | 10.87 | 4.54 | 2.39x |
| 8 | ldc_2d | 21.40 | 9.55 | 2.24x |
| 9 | ode_spring_mass | 17.78 | 4.97 | 3.58x |
| 10 | seismic_wave | 34.62 | 16.12 | 2.15x |
| 11 | surface_pde/sphere | 15.88 | 4.17 | 3.81x |
| 12 | taylor_green | 61.67 | 12.42 | 4.97x |
| 13 | three_fin_2d | 57.42 | 21.97 | 2.61x |
| 14 | wave_1d | 18.51 | 5.27 | 3.51x |
| 15 | wave_1d_causal | 15.48 | 5.67 | 2.73x |
| 16 | wave_inverse | 13.41 | 6.31 | 2.13x |
| 17 | waveguide/cavity_2D | 33.02 | 17.17 | 1.92x |
| 18 | waveguide/slab_2D | 34.00 | 16.82 | 2.02x |

**统计摘要：**
- 平均加速比：**2.66x**
- 最大加速比：4.97x (taylor_green, 3D NS时间相关)
- 最小加速比：1.16x (anti_derivative, DeepONet数据驱动)
- 所有18个模型 CINN 均有加速

### 7.2 性能分析

| 模型类型 | 典型加速比 | 原因分析 |
|---------|-----------|---------|
| 3D+时间相关 (taylor_green) | 4.97x | 计算图最复杂，CINN 融合收益最大 |
| ODE/简单PDE (spring_mass, sphere, wave_1d) | 3.5-3.8x | 算子数多但 shape 小，融合消除 kernel launch 开销 |
| 2D NS (annular, cylinder, ldc, chip) | 2.2-2.9x | 中等复杂度，稳定加速 |
| 弹性力学 (bracket, panel) | 1.8-2.7x | 多约束+大 batch，IO bound 成分增加 |
| 波导/高频 (waveguide) | 1.9-2.0x | modified_fourier arch，部分算子未融合 |
| 数据驱动 (anti_derivative) | 1.16x | DeepONet 结构简单，优化空间有限 |

### 7.3 精度对比（全部18模型，total loss，50 iters 平均）

| # | 模型 | 动态图 total loss | CINN total loss | 相对误差 |
|---|------|------------------|-----------------|----------|
| 1 | annular_ring | 4.7910126874 | 4.7908422521 | 0.004% |
| 2 | anti_derivative | 1810.0900048828 | 1810.0901586914 | 0.000009% |
| 3 | bracket | 18.5356299778 | 18.5356147513 | 0.0001% |
| 4 | chip_2d | 5.5324703560 | 5.5324231242 | 0.0009% |
| 5 | cylinder_2d | 5.0641854516 | 5.0641788751 | 0.0001% |
| 6 | fuselage_panel | 57.9197781527 | 57.9190531259 | 0.001% |
| 7 | helmholtz | 10062.0106177223 | 10062.0105395395 | 0.000001% |
| 8 | ldc_2d | 0.0501167617 | 0.0501167467 | 0.00003% |
| 9 | ode_spring_mass | 1.6599861923 | 1.6599564856 | 0.002% |
| 10 | seismic_wave | 2449.7970303632 | 2449.7969767572 | 0.000002% |
| 11 | surface_pde/sphere | 51.9551512682 | 51.9551749033 | 0.00005% |
| 12 | taylor_green | 6668.9862074170 | 6668.9816690197 | 0.00007% |
| 13 | three_fin_2d | 2.0785426980 | 2.0784973095 | 0.002% |
| 14 | wave_1d | 3.3015540230 | 3.3015553114 | 0.00004% |
| 15 | wave_1d_causal | 180.5413227573 | 180.5452699464 | 0.002% |
| 16 | wave_inverse | 4388.9892043495 | 4389.0011499882 | 0.0003% |
| 17 | waveguide/cavity_2D | 2179033.5331167602 | 2179031.7557720942 | 0.00008% |
| 18 | waveguide/slab_2D | 1979717.0759176635 | 1979714.9348419190 | 0.0001% |

**精度结论：**
- 所有18个模型 total loss 相对误差均 < 0.005%
- 最大相对误差 0.004% (annular_ring)，大部分模型 < 0.001%
- CINN 编译器优化未引入任何可观测的精度损失
- 差异来源于浮点运算顺序变化（算子融合后 reduction 顺序不同），属于正常数值波动

### 7.4 测试命令

全量批量测试：
```bash
cd /work/modulus-sym
bash run_all_infer_tests.sh
```

单模型测试示例：
```bash
# 动态图
bash test_tipc/dynamic/helmholtz-helmholtz/N1C1/helmholtz-helmholtz_bs1_fp32_DP_infer.sh

# 动转静+CINN
bash test_tipc/dynamicTostatic/helmholtz-helmholtz/N1C1/helmholtz-helmholtz_bs1_fp32_DP_infer.sh
```

查看结果：
```bash
grep "time/iteration:" modulus-sym_*_infer_log modulus-sym_*_d2sT_infer_log
```

## 8. 测试模型列表（全部41个完成）

### 8.1 原有18个模型

| # | 模型目录名 | 推理脚本 | 状态 |
|---|-----------|---------|------|
| 1 | annular_ring-annular_ring-annular_ring | `examples/annular_ring/annular_ring/annular_ring_infer.py` | ✅ |
| 2 | anti_derivative-data_informed | `examples/anti_derivative/data_informed_infer.py` | ✅ |
| 3 | bracket-bracket | `examples/bracket/bracket_infer.py` | ✅ |
| 4 | chip_2d-chip_2d | `examples/chip_2d/chip_2d_infer.py` | ✅ |
| 5 | cylinder-cylinder_2d | `examples/cylinder/cylinder_2d_infer.py` | ✅ |
| 6 | fuselage_panel-panel | `examples/fuselage_panel/panel_infer.py` | ✅ |
| 7 | helmholtz-helmholtz | `examples/helmholtz/helmholtz_infer.py` | ✅ |
| 8 | ldc-ldc_2d | `examples/ldc/ldc_2d_infer.py` | ✅ |
| 9 | ode_spring_mass-spring_mass_solver | `examples/ode_spring_mass/spring_mass_solver_infer.py` | ✅ |
| 10 | seismic_wave-wave_2d | `examples/seismic_wave/wave_2d_infer.py` | ✅ |
| 11 | surface_pde-sphere-sphere | `examples/surface_pde/sphere/sphere_infer.py` | ✅ |
| 12 | taylor_green-taylor_green | `examples/taylor_green/taylor_green_infer.py` | ✅ |
| 13 | three_fin_2d-heat_sink | `examples/three_fin_2d/heat_sink_infer.py` | ✅ |
| 14 | wave_equation-wave_1d | `examples/wave_equation/wave_1d_infer.py` | ✅ |
| 15 | wave_equation-wave_1d_causal | `examples/wave_equation/wave_1d_causal_infer.py` | ✅ |
| 16 | wave_equation-wave_inverse | `examples/wave_equation/wave_inverse_infer.py` | ✅ |
| 17 | waveguide-cavity_2D-waveguide2D_TMz | `examples/waveguide/cavity_2D/waveguide2D_TMz_infer.py` | ✅ |
| 18 | waveguide-slab_2D-slab_2D | `examples/waveguide/slab_2D/slab_2D_infer.py` | ✅ |

### 8.2 新补全23个模型（2026-06-29）

| # | 模型目录名 | 推理脚本 | 状态 |
|---|-----------|---------|------|
| 19 | ldc-ldc_2d_importance_sampling | `examples/ldc/ldc_2d_importance_sampling_infer.py` | ✅ |
| 20 | ldc-ldc_2d_domain_decomposition | `examples/ldc/ldc_2d_domain_decomposition_infer.py` | ✅ |
| 21 | ldc-ldc_2d_domain_decomposition_fbpinn | `examples/ldc/ldc_2d_domain_decomposition_fbpinn_infer.py` | ✅ |
| 22 | ldc-ldc_2d_zeroEq | `examples/ldc/ldc_2d_zeroEq_infer.py` | ✅ |
| 23 | chip_2d-chip_2d_solid_solid_heat_transfer | `examples/chip_2d/chip_2d_solid_solid_heat_transfer_infer.py` | ✅ |
| 24 | chip_2d-chip_2d_solid_fluid_heat_transfer_heat | `examples/chip_2d/chip_2d_solid_fluid_heat_transfer_heat_infer.py` | ✅ |
| 25 | chip_2d-chip_2d_solid_fluid_heat_transfer_flow | `examples/chip_2d/chip_2d_solid_fluid_heat_transfer_flow_infer.py` | ✅ |
| 26 | annular_ring-annular_ring_parameterized-annular_ring_parameterized | `examples/annular_ring/annular_ring_parameterized/annular_ring_parameterized_infer.py` | ✅ |
| 27 | annular_ring-annular_ring_equation_instancing-annular_ring | `examples/annular_ring/annular_ring_equation_instancing/annular_ring_infer.py` | ✅ |
| 28 | annular_ring-annular_ring_gradient_enhanced-annular_ring_gradient_enhanced | `examples/annular_ring/annular_ring_gradient_enhanced/annular_ring_gradient_enhanced_infer.py` | ✅ |
| 29 | limerock-limerock_hFTB-limerock_thermal | `examples/limerock/limerock_hFTB/limerock_thermal_infer.py` | ✅ |
| 30 | taylor_green-taylor_green_causal | `examples/taylor_green/taylor_green_causal_infer.py` | ✅ |
| 31 | three_fin_2d-heat_sink_inverse | `examples/three_fin_2d/heat_sink_inverse_infer.py` | ✅ |
| 32 | three_fin_3d-three_fin_flow | `examples/three_fin_3d/three_fin_flow_infer.py` | ✅ |
| 33 | three_fin_3d-three_fin_thermal | `examples/three_fin_3d/three_fin_thermal_infer.py` | ✅ |
| 34 | waveguide-cavity_3D-waveguide3D | `examples/waveguide/cavity_3D/waveguide3D_infer.py` | ✅ |
| 35 | waveguide-slab_3D-slab_3D | `examples/waveguide/slab_3D/slab_3D_infer.py` | ✅ |
| 36 | turbulent_channel-2d_std_wf-u_tau_lookup | `examples/turbulent_channel/2d_std_wf/u_tau_lookup_infer.py` | ✅ |
| 37 | turbulent_channel-2d-re590_k_om_LS | `examples/turbulent_channel/2d/re590_k_om_LS_infer.py` | ✅ |
| 38 | turbulent_channel-2d-re590_k_ep_LS | `examples/turbulent_channel/2d/re590_k_ep_LS_infer.py` | ✅ |
| 39 | turbulent_channel-2d_std_wf-re590_k_ep | `examples/turbulent_channel/2d_std_wf/re590_k_ep_infer.py` | ✅ |
| 40 | turbulent_channel-2d_std_wf-re590_k_om | `examples/turbulent_channel/2d_std_wf/re590_k_om_infer.py` | ✅ |
| 41 | anti_derivative-physics_informed | `examples/anti_derivative/physics_informed_infer.py` | ✅ |


## modulus目录下模型合集
序号	案例名称
1	ldc/ldc_2d
2	chip_2d/chip_2d_solid_solid_heat_transfer
3	turbulent_channel/2d_std_wf/u_tau_lookup
4	turbulent_channel/2d/re590_k_om_LS
5	three_fin_2d/heat_sink
6	three_fin_3d/three_fin_flow
7	chip_2d/chip_2d
8	three_fin_2d/heat_sink_inverse
9	chip_2d/chip_2d_solid_fluid_heat_transfer_heat
10	turbulent_channel/2d/re590_k_ep_LS
11	turbulent_channel/2d_std_wf/re590_k_ep
12	turbulent_channel/2d_std_wf/re590_k_om
13	three_fin_3d/three_fin_thermal
14	ode_spring_mass/spring_mass_solver
15	wave_equation/wave_1d
16	ldc/ldc_2d_importance_sampling
17	helmholtz/helmholtz
18	ldc/ldc_2d_domain_decomposition
19	surface_pde/sphere/sphere
20	ldc/ldc_2d_domain_decomposition_fbpinn
21	seismic_wave/wave_2d
22	wave_equation/wave_1d_causal
23	chip_2d/chip_2d_solid_fluid_heat_transfer_flow
24	wave_equation/wave_inverse
25	limerock/limerock_hFTB/limerock_thermal
26	annular_ring/annular_ring_parameterized/annular_ring_parameterized
27	annular_ring/annular_ring_equation_instancing/annular_ring
28	annular_ring/annular_ring/annular_ring
29	cylinder/cylinder_2d
30	taylor_green/taylor_green_causal
31	taylor_green/taylor_green
32	annular_ring/annular_ring_gradient_enhanced/annular_ring_gradient_enhanced
33	waveguide/slab_2D/slab_2D
34	waveguide/cavity_2D/waveguide2D_TMz
35	waveguide/cavity_3D/waveguide3D
36	waveguide/slab_3D/slab_3D
37	ldc/ldc_2d_zeroEq
38	bracket/bracket
39	fuselage_panel/panel
40	anti_derivative/data_informed
41	anti_derivative/physics_informed

## 9. 全量41模型推理性能结果（2026-06-29，A100-80GB）

### 9.1 本轮测试配置

- 测试范围：41 个科学计算模型，其中包含本轮新增补全的 23 个推理模型。
- 测试模式：动态图（算子库） vs 动转静 + CINN。
- 环境变量：`INFER_WARMUP=10`，`INFER_STEPS=100`，`debug=1`，`loss_monitor=1`。
- `debug=1` 用于关闭数据 shuffle / worker 随机性，`loss_monitor=1` 用于避免 debug 模式加载固定 init ckpt，二者配合用于稳定 loss 对比。
- 原始日志：`/work/modulus-sym/modulus-sym_*_N1C1_infer_log` 和 `/work/modulus-sym/modulus-sym_*_N1C1_d2sT_infer_log`。

### 9.2 指标说明

- `time/iteration:` 是本次性能测试的主指标，单位 ms/iteration。
- `run_mode: DP` 表示 benchmark 脚本沿用 TIPC 命名里的 Data Parallel 单卡模式；本次 `device_num=N1C1`，实际是单机单卡。
- `model_run_time` 是单个 shell benchmark 进程的墙钟耗时（脚本用 `date +%Y%m%d%H%M%S` 做整数相减），包含模型初始化、checkpoint 处理、CINN 编译、warmup、benchmark 和日志解析开销，不等同于单步延迟。
- `ips: NaN` 是 `analysis_log.py` 对 `ms/iteration` 类型日志未正确换算出的派生字段；本轮以日志中的 `time/iteration:` 为准，benchmark 本身已成功输出性能数据。
- loss 默认不在 `new23_infer_run.log` 的 `tail -5` 汇总中展示；精度 / loss 需要查看对应 `*_infer_log` 内的 `[Accuracy]` 段或单独开启全量精度测试汇总。

### 9.3 41模型性能对比（time/iteration, ms）

| # | 模型 | 动态图(ms) | 动转静+CINN(ms) | 加速比 |
|---|------|-----------:|----------------:|-------:|
| 1 | annular_ring/annular_ring/annular_ring | 28.89 | 10.70 | 2.70x |
| 2 | anti_derivative/data_informed | 17.21 | 14.80 | 1.16x |
| 3 | bracket/bracket | 54.13 | 19.78 | 2.74x |
| 4 | chip_2d/chip_2d | 47.20 | 16.08 | 2.94x |
| 5 | cylinder/cylinder_2d | 27.76 | 11.24 | 2.47x |
| 6 | fuselage_panel/panel | 43.82 | 24.99 | 1.75x |
| 7 | helmholtz/helmholtz | 22.30 | 4.54 | 4.92x |
| 8 | ldc/ldc_2d | 21.40 | 9.55 | 2.24x |
| 9 | ode_spring_mass/spring_mass_solver | 17.78 | 4.97 | 3.58x |
| 10 | seismic_wave/wave_2d | 34.62 | 16.12 | 2.15x |
| 11 | surface_pde/sphere/sphere | 15.88 | 4.17 | 3.81x |
| 12 | taylor_green/taylor_green | 61.67 | 12.42 | 4.96x |
| 13 | three_fin_2d/heat_sink | 57.42 | 21.97 | 2.61x |
| 14 | wave_equation/wave_1d | 18.51 | 5.27 | 3.51x |
| 15 | wave_equation/wave_1d_causal | 15.48 | 5.67 | 2.73x |
| 16 | wave_equation/wave_inverse | 13.41 | 6.31 | 2.13x |
| 17 | waveguide/cavity_2D/waveguide2D_TMz | 33.02 | 17.17 | 1.92x |
| 18 | waveguide/slab_2D/slab_2D | 34.00 | 16.82 | 2.02x |
| 19 | ldc/ldc_2d_importance_sampling | 28.90 | 26.57 | 1.09x |
| 20 | ldc/ldc_2d_domain_decomposition | 40.30 | 41.71 | 0.97x |
| 21 | ldc/ldc_2d_domain_decomposition_fbpinn | 42.18 | 43.33 | 0.97x |
| 22 | ldc/ldc_2d_zeroEq | 29.67 | 29.58 | 1.00x |
| 23 | chip_2d/chip_2d_solid_solid_heat_transfer | 90.35 | 105.04 | 0.86x |
| 24 | chip_2d/chip_2d_solid_fluid_heat_transfer_heat | 113.46 | 115.55 | 0.98x |
| 25 | chip_2d/chip_2d_solid_fluid_heat_transfer_flow | 51.23 | 48.44 | 1.06x |
| 26 | annular_ring/annular_ring_parameterized/annular_ring_parameterized | 27.14 | 27.60 | 0.98x |
| 27 | annular_ring/annular_ring_equation_instancing/annular_ring | 27.33 | 26.11 | 1.05x |
| 28 | annular_ring/annular_ring_gradient_enhanced/annular_ring_gradient_enhanced | 75.32 | 73.93 | 1.02x |
| 29 | limerock/limerock_hFTB/limerock_thermal | 59.11 | 61.03 | 0.97x |
| 30 | taylor_green/taylor_green_causal | 51.09 | 50.68 | 1.01x |
| 31 | three_fin_2d/heat_sink_inverse | 23.89 | 23.70 | 1.01x |
| 32 | three_fin_3d/three_fin_flow | 207.66 | 209.38 | 0.99x |
| 33 | three_fin_3d/three_fin_thermal | 69.16 | 73.02 | 0.95x |
| 34 | waveguide/cavity_3D/waveguide3D | 115.09 | 85.32 | 1.35x |
| 35 | waveguide/slab_3D/slab_3D | 133.96 | 119.05 | 1.13x |
| 36 | turbulent_channel/2d_std_wf/u_tau_lookup | 5.56 | 5.55 | 1.00x |
| 37 | turbulent_channel/2d/re590_k_om_LS | 88.49 | 81.60 | 1.08x |
| 38 | turbulent_channel/2d/re590_k_ep_LS | 90.71 | 91.36 | 0.99x |
| 39 | turbulent_channel/2d_std_wf/re590_k_ep | 94.17 | 93.13 | 1.01x |
| 40 | turbulent_channel/2d_std_wf/re590_k_om | 90.62 | 82.55 | 1.10x |
| 41 | anti_derivative/physics_informed | 30.32 | 29.67 | 1.02x |

### 9.4 统计摘要

- 41 个模型均完成动态图和动转静+CINN推理性能测试，均成功输出 `time/iteration:`。
- 41 模型平均加速比：1.70x。
- 原 18 个模型平均加速比：2.63x，CINN 加速收益明显。
- 新增 23 个模型平均加速比：0.98x，整体接近持平；其中 `waveguide/cavity_3D/waveguide3D` 有 1.35x 加速，部分 heat-transfer / domain-decomposition / turbulent-channel 模型接近持平或轻微回退。
- 推测新增模型收益较低的主要原因：约束/几何采样/数据加载占比更高，计算图可融合部分占比低于原 18 个典型 PINN 模型。

### 9.5 本轮修复记录

- 修复 `examples/taylor_green/taylor_green_causal_infer.py` 中残留的空 `if cfg.training.max_steps <= 600:` 分支导致的 `IndentationError`。
- 为 82 个 `benchmark_common_infer/run_benchmark.sh` 统一添加 `LD_LIBRARY_PATH=/usr/lib64:${LD_LIBRARY_PATH}`，解决 `libcuda.so.1` 查找失败。
- 为 82 个 runner 统一添加 `debug=1` 和 `loss_monitor=1`，用于稳定 loss 数据并避免随机采样影响精度对比。
- `run_all_infer_tests.sh` 与 `run_new23_infer_tests.sh` 顶层也已显式导出 `debug=1` / `loss_monitor=1`。
