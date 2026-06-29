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

## 8. 测试模型列表（全部完成）

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
