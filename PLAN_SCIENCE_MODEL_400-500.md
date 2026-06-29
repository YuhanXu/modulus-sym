export MODEL_REPO_ROOT=/work/PaddleX/
export BENCHMARK_ROOT=/work/tools/;
也需要在(env3.10) 的虚拟环境里执行

1、modulus-sym模型运行方式
下载modulus-sym库
git clone http://github.com/PaddleBenchmark/modulus-sym.git -b modified_paddle_dy2st && cd modulus-sym && git checkout -b b005a42dcbb587654c8a909102995c4f212ea8d6 b005a42dcbb587654c8a909102995c4f212ea8d6;
export PROFILING_TIMER_ONLY=no;
运行模型脚本
#动态图模型
export FLAGS_set_to_1d=0;export CUDA_MODULE_LOADING=LAZY;export ENABLE_CINN_IN_DY2ST=0
bash test_tipc/dynamic/annular_ring-annular_ring-annular_ring/N1C1/annular_ring-annular_ring-annular_ring_bs1_fp32_DP.sh;

#动转静模型
export ENABLE_CINN_IN_DY2ST=1;export  FLAGS_enable_auto_recompute=1;export FLAGS_prim_vjp_skip_default_ops=False;export FLAGS_cinn_debug=1
bash test_tipc/dynamicTostatic/annular_ring-annular_ring-annular_ring/N1C1/annular_ring-annular_ring-annular_ring_bs1_fp32_DP.sh

2、deepmd模型运行方式
下载deepmd库
git clone https://github.com/deepmodeling/deepmd-kit -b master && cd deepmd-kit && git checkout -b 24e54bfb44c18f96b012035c757a5d9be3d1fa73 24e54bfb44c18f96b012035c757a5d9be3d1fa73;
cp -r ../benchmark/frame_benchmark/paddle/deepmd-kit_train_benchmark  ./deepmd-kit/train_benchmark
export PROFILING_TIMER_ONLY=no;

运行模型脚本
#deepmd模型动态图和静态图是一个脚本，区别就是环境变量
#动态图环境变量
export CINN=1;export ENABLE_CINN_IN_DY2ST=1;export FLAGS_cinn_debug=1;export CINN_ALLOW_DYNAMIC_SHAPE=0
bash train_benchmark/dynamic/dpa2/N1C1/dpa2_bs1_fp64_DP.sh;


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

跑科学计算模型生成的前500step的log文件保存到/work/PaddleX_QA_test_new_science_400-500目录下，生成的400-500step开CINN和动态图的收敛性精度对比图也放在work/PaddleX_QA_test_new_science_400-500目录下

有任何关键进展或者报错，请梳理之后更新到/work/PaddleX_QA_test_new_400-500/PLAN_SCIENCE_MODEL_400-500.md这个文件里。

---

## 执行进展（2026-06-22）

### 重要问题确认：当前第一批 loss delta 图不能作为严格结论

用户质疑 delta loss outliers 数量异常偏多后，检查运行日志和脚本发现关键干扰因素：

1. `run_science_models.py` 中 dy 和 CINN 虽然环境变量不同，且 CINN 日志确认出现 `Using jit.to_static`，说明 CINN/动转静确实生效；
2. 但 dy 和 CINN 使用了相同的 Hydra override，因此输出目录相同，例如：
   - `outputs/training.max_steps=500,training.print_stats_freq=1,training.rec_inference_freq=9999,training.rec_monitor_freq=9999,training.rec_validation_freq=9999,training.save_network_freq=9999/annular_ring`
3. dy 和 CINN 日志均出现：`attempting to restore from: outputs/...`，说明训练可能从已有 checkpoint 恢复；
4. 因 dy/CINN 输出目录未隔离，且多次运行后 checkpoint 可能互相复用或覆盖，step 400-500 的 loss 对比不一定来自同一起点训练，因此大量 delta loss outliers 可能被 checkpoint restore 干扰放大。

补充偏差：第一批运行没有直接调用 PLAN 中的 `test_tipc/dynamic/.../N1C1/*.sh` 和 `test_tipc/dynamicTostatic/.../N1C1/*.sh` wrapper，而是从 `benchmark_common/run_benchmark.sh` 中抽取 `train_cmd` 后手动设置环境变量执行；虽然 CINN 日志确认 `jit.to_static` 生效，但这仍不是严格的 wrapper 原样执行方式。

修正计划：
- 修改运行脚本，优先按 PLAN 的 dynamic / dynamicTostatic 两套目录语义执行，或至少读取对应目录脚本并补齐 wrapper 中的环境；
- 为 dy 和 CINN 增加不同的 Hydra 输出目录（例如 `hydra.run.dir=outputs_science_400_500/<model>/dy` 和 `.../cinn`），或在每次运行前清理对应模型输出目录；
- 优先选取代表模型重跑验证：例如 `annular_ring/annular_ring`、`chip_2d/chip_2d`、`helmholtz/helmholtz`、`wave_equation/wave_inverse`、`ldc/ldc_2d`；
- 代表模型验证后，再决定是否全量重跑41个模型并重新生成 loss 对比图；
- 当前已生成的41张图保留为“第一批未隔离 checkpoint / 未开启 debug+loss_monitor 的结果”，不作为最终精度结论。

### debug=1 + loss_monitor=1 代表模型验证

按用户建议，为 dy 和 CINN 均增加环境变量：

```bash
export debug=1
export loss_monitor=1
```

修改位置：`/work/PaddleX_QA_test_new_science_400-500/run_science_models.py`

同时新增代表模型验证脚本：`/work/PaddleX_QA_test_new_science_400-500/run_debug_loss_monitor_subset.py`
- 代表模型：
  - `annular_ring/annular_ring/annular_ring`（第一批 outliers 多）
  - `ldc/ldc_2d`（第一批 PASS，对照）
- dy/CINN 分别使用独立 `hydra.run.dir`，避免 checkpoint 污染
- dy/CINN 均跑 500 step

验证结果：

| 模型 | dy step0 loss | CINN step0 loss | CINN to_static | 400-500 outliers | 结论 |
|------|---------------|-----------------|----------------|------------------|------|
| annular_ring/annular_ring/annular_ring | 5.7862143517 | 5.7859244347 | True | 88/101 | debug/loss_monitor 使初始化一致，但后续训练轨迹仍明显不同 |
| ldc/ldc_2d | 0.0498795696 | 0.0498795509 | True | 1/101 | 基本一致，验证 debug/loss_monitor 对随机初始化问题有效 |

生成的验证图：
- `/work/PaddleX_QA_test_new_science_400-500/annular_ring-annular_ring-annular_ring_debug_loss_400-500.png`
- `/work/PaddleX_QA_test_new_science_400-500/ldc-ldc_2d_debug_loss_400-500.png`

阶段结论：
1. `debug=1` + `loss_monitor=1` 对“dy/CINN step0 loss 随机不一致”有效；
2. 但并不能保证所有模型 step 400-500 完全收敛一致，`annular_ring` 仍有 88/101 outliers；
3. 因此后续全量重跑应开启这两个环境变量，并保留独立输出目录；对仍 FAIL 的模型需要进一步排查 CINN 计算/训练轨迹差异。

### 阶段一：41个模型训练完成

执行脚本：`/work/PaddleX_QA_test_new_science_400-500/run_science_models.py`

全部41个模型均跑完500步（dy + CINN），结果 **40/41 PASS**。

唯一训练阶段 FAIL：`limerock/limerock_hFTB/limerock_thermal`
- 原因：动态图（dy）在1800秒超时前未能跑完500步，CINN正常完成500步
- 日志：`limerock-limerock_hFTB-limerock_thermal_dy.log`，`limerock-limerock_hFTB-limerock_thermal_CINN.log`
- 处理：按用户确认，该模型不再继续重跑，保留当前失败记录；loss对比图仍按现有日志生成

### 阶段二：loss对比图批量生成完成

执行脚本：`/work/PaddleX_QA_test_new_science_400-500/plot_400-500steps.py`

生成图片目录：`/work/PaddleX_QA_test_new_science_400-500/`，命名规则：`<model>_loss_400-500.png`
完整汇总：`/work/PaddleX_QA_test_new_science_400-500/science_loss_400-500_summary.txt`

**判断标准**：step 400-500 内，|loss_CINN - loss_dy| > 0.01 的 outlier 步数 ≤ 10 为 PASS，否则为 FAIL

生成图数：41，PASS=11，FAIL=30

#### PASS 模型（11个，outliers≤10）

| 模型 | outliers |
|------|---------|
| ldc/ldc_2d | 0 |
| ldc/ldc_2d_domain_decomposition | 0 |
| ldc/ldc_2d_domain_decomposition_fbpinn | 0 |
| ldc/ldc_2d_importance_sampling | 0 |
| ode_spring_mass/spring_mass_solver | 4 |
| three_fin_3d/three_fin_thermal | 0 |
| turbulent_channel/2d/re590_k_ep_LS | 0 |
| turbulent_channel/2d/re590_k_om_LS | 0 |
| turbulent_channel/2d_std_wf/re590_k_ep | 0 |
| turbulent_channel/2d_std_wf/re590_k_om | 0 |
| wave_equation/wave_1d | 0 |

#### FAIL 模型（30个，outliers>10）

| 模型 | outliers/101 |
|------|-------------|
| annular_ring/annular_ring | 59 |
| annular_ring/annular_ring_equation_instancing | 52 |
| annular_ring/annular_ring_gradient_enhanced | 46 |
| annular_ring/annular_ring_parameterized | 79 |
| anti_derivative/data_informed | 101 |
| anti_derivative/physics_informed | 101 |
| bracket/bracket | 73 |
| chip_2d/chip_2d | 87 |
| chip_2d/chip_2d_solid_fluid_heat_transfer_flow | 81 |
| chip_2d/chip_2d_solid_fluid_heat_transfer_heat | 101 |
| chip_2d/chip_2d_solid_solid_heat_transfer | 101 |
| cylinder/cylinder_2d | 29 |
| fuselage_panel/panel | 101 |
| helmholtz/helmholtz | 101 |
| ldc/ldc_2d_zeroEq | 13 |
| limerock/limerock_hFTB/limerock_thermal | 81 (dy日志不完整) |
| seismic_wave/wave_2d | 101 |
| surface_pde/sphere/sphere | 29 |
| taylor_green/taylor_green | 101 |
| taylor_green/taylor_green_causal | 101 |
| three_fin_2d/heat_sink | 14 |
| three_fin_2d/heat_sink_inverse | 101 |
| three_fin_3d/three_fin_flow | 99 |
| turbulent_channel/2d_std_wf/u_tau_lookup | 101 |
| wave_equation/wave_1d_causal | 35 |
| wave_equation/wave_inverse | 101 |
| waveguide/cavity_2D/waveguide2D_TMz | 101 |
| waveguide/cavity_3D/waveguide3D | 101 |
| waveguide/slab_2D/slab_2D | 101 |
| waveguide/slab_3D/slab_3D | 101 |

---

## 全量重跑结果（debug=1 + loss_monitor=1，隔离输出目录）

### 执行配置

按用户要求，已删除 `/work/PaddleX_QA_test_new_science_400-500/` 下旧的 `.log` 和 `.png` 文件，保留脚本文件，然后重新批跑 41 个科学计算模型。

本次全量重跑使用：

```bash
export debug=1
export loss_monitor=1
```

并修正执行脚本：`/work/PaddleX_QA_test_new_science_400-500/run_science_models.py`

关键修正：
- dy 从 `test_tipc/dynamic` 读取运行命令；
- CINN 从 `test_tipc/dynamicTostatic` 读取运行命令；
- dy/CINN 分别使用独立 `hydra.run.dir`：`outputs_science_400_500_debug_loss_monitor/<model>/<mode>`，避免 checkpoint 污染；
- dy/CINN 均开启 `debug=1`、`loss_monitor=1`；
- 每个模型均设置 `training.max_steps=500` 和 `training.print_stats_freq=1`，保证每 step 输出 loss。

### 训练阶段结果

41 个模型全部完成 dy + CINN 500 step 训练：**41/41 PASS**。

日志输出目录：`/work/PaddleX_QA_test_new_science_400-500/`

### 400-500 step loss 对比图结果

执行脚本：`/work/PaddleX_QA_test_new_science_400-500/plot_400-500steps.py`

汇总文件：`/work/PaddleX_QA_test_new_science_400-500/science_loss_400-500_summary.txt`

判断标准：step 400-500 内，`|loss_CINN - loss_dy| > 0.01` 的 outlier 步数 ≤ 10 为 PASS，否则为 FAIL。

生成图数：41，PASS=18，FAIL=23，SKIP=0。

#### PASS 模型（18个）

| 模型 | outliers/101 |
|------|-------------|
| bracket/bracket | 0 |
| chip_2d/chip_2d_solid_fluid_heat_transfer_flow | 3 |
| ldc/ldc_2d | 1 |
| ldc/ldc_2d_domain_decomposition | 0 |
| ldc/ldc_2d_domain_decomposition_fbpinn | 0 |
| ldc/ldc_2d_importance_sampling | 0 |
| ldc/ldc_2d_zeroEq | 0 |
| limerock/limerock_hFTB/limerock_thermal | 0 |
| ode_spring_mass/spring_mass_solver | 7 |
| seismic_wave/wave_2d | 7 |
| three_fin_2d/heat_sink | 6 |
| three_fin_3d/three_fin_thermal | 0 |
| turbulent_channel/2d/re590_k_ep_LS | 0 |
| turbulent_channel/2d/re590_k_om_LS | 0 |
| turbulent_channel/2d_std_wf/re590_k_ep | 0 |
| turbulent_channel/2d_std_wf/re590_k_om | 0 |
| wave_equation/wave_1d | 4 |
| wave_equation/wave_1d_causal | 2 |

#### FAIL 模型（23个）

| 模型 | outliers/101 |
|------|-------------|
| annular_ring/annular_ring | 88 |
| annular_ring/annular_ring_equation_instancing | 101 |
| annular_ring/annular_ring_gradient_enhanced | 42 |
| annular_ring/annular_ring_parameterized | 46 |
| anti_derivative/data_informed | 67 |
| anti_derivative/physics_informed | 101 |
| chip_2d/chip_2d | 79 |
| chip_2d/chip_2d_solid_fluid_heat_transfer_heat | 101 |
| chip_2d/chip_2d_solid_solid_heat_transfer | 101 |
| cylinder/cylinder_2d | 27 |
| fuselage_panel/panel | 67 |
| helmholtz/helmholtz | 100 |
| surface_pde/sphere/sphere | 23 |
| taylor_green/taylor_green | 100 |
| taylor_green/taylor_green_causal | 101 |
| three_fin_2d/heat_sink_inverse | 101 |
| three_fin_3d/three_fin_flow | 51 |
| turbulent_channel/2d_std_wf/u_tau_lookup | 101 |
| wave_equation/wave_inverse | 101 |
| waveguide/cavity_2D/waveguide2D_TMz | 100 |
| waveguide/cavity_3D/waveguide3D | 76 |
| waveguide/slab_2D/slab_2D | 101 |
| waveguide/slab_3D/slab_3D | 98 |

### 本轮结论

1. 加入 `debug=1` 和 `loss_monitor=1` 后，训练阶段稳定性明显提升：41 个模型均完成 dy/CINN 500 step，之前超时的 `limerock_thermal` 也完成并且 loss 对比 PASS（outliers=0）。
2. 隔离 `hydra.run.dir` 后，checkpoint 污染问题已规避。
3. 400-500 step loss 对比结果由第一批的 11 PASS / 30 FAIL 改善为 18 PASS / 23 FAIL。
4. 仍 FAIL 的 23 个模型需要进一步排查 CINN 计算差异或训练轨迹差异。

---

## 两个科学计算模型全量训练（不覆盖 training.max_steps）

### 执行目标

按用户要求，选择两个模型做单卡全量训练（不配置 `training.max_steps`，走模型配置默认值），日志保存到：

`/work/PaddleX_QA_test_new_science_max_step/`

模型与默认训练步数：

| 模型 | 配置文件默认 max_steps |
|------|------------------------|
| ldc/ldc_2d | 10000 |
| annular_ring/annular_ring | 200000 |

执行脚本：`/work/PaddleX_QA_test_new_science_max_step/run_full_training.py`

环境变量：dy/CINN 均开启：

```bash
export debug=1
export loss_monitor=1
```

并使用独立 `hydra.run.dir=outputs_science_max_step/<model>/<mode>` 避免 checkpoint 污染。

### 当前训练进展（2026-06-22）

日志文件：

- `/work/PaddleX_QA_test_new_science_max_step/ldc-ldc_2d_dy.log`
- `/work/PaddleX_QA_test_new_science_max_step/ldc-ldc_2d_cinn.log`
- `/work/PaddleX_QA_test_new_science_max_step/annular_ring-annular_ring-annular_ring_dy.log`
- `/work/PaddleX_QA_test_new_science_max_step/annular_ring-annular_ring-annular_ring_cinn.log`

训练完成情况：

| 模型 | dy | CINN | 状态 |
|------|----|------|------|
| ldc/ldc_2d | 10000/10000 | 10000/10000 | 已完成 |
| annular_ring/annular_ring | 200000/200000 | 106092/200000（仍在运行） | CINN 未完成 |

当前仍在运行的进程：`annular_ring.py` CINN，全量 200000 step 训练中。

### 已生成的 loss 收敛对比图

绘图脚本：`/work/PaddleX_QA_test_new_science_max_step/plot_full_training.py`

已生成：

- `/work/PaddleX_QA_test_new_science_max_step/ldc-ldc_2d_full_loss_compare.png`
  - dy_steps=10001，max_dy=10000
  - cinn_steps=10001，max_cinn=10000
  - common_steps=10001
  - outliers(|delta|>1e-2)=8/10001
- `/work/PaddleX_QA_test_new_science_max_step/annular_ring-annular_ring-annular_ring_partial_1-106092_loss_compare.png`
  - dy_steps=200001，max_dy=200000
  - cinn_steps=106093，max_cinn=106092
  - common_steps=106093
  - outliers(|delta|>1e-2)=59830/106093
  - 注：CINN 已完成全量 200000 step，已重新生成 full 图（下见）。

- `/work/PaddleX_QA_test_new_science_max_step/annular_ring-annular_ring-annular_ring_full_loss_compare.png`（全量图）
  - dy_steps=200001，max_dy=200000
  - cinn_steps=200001，max_cinn=200000
  - common_steps=200001
  - outliers(|delta|>1e-2)=64785/200001（约 32%）

### annular_ring loss 暴增的初步判断

从当前全量训练日志看，dy 和 CINN 均开启了 `debug=1`、`loss_monitor=1` 且使用独立输出目录，初始化不一致和 checkpoint 污染已不是主要嫌疑。

`annular_ring/annular_ring` 在 500 step 验证中 step0 loss 已接近一致，但 400-500 区间仍有大量 outliers；本次全量训练（200000 step）完成后 outliers=64785/200001（约32%），说明更可能是 CINN 动转静后某些计算路径/训练轨迹逐步偏离，而不是简单的初始值不同。

### waveguide/cavity_3D/waveguide3D 全量训练（进行中）

执行脚本：`/work/PaddleX_QA_test_new_science_max_step/run_waveguide3d_full_training.py`

默认 max_steps=500000。

当前状态（2026-06-23）：

- dy：任务中断，日志最后到 step 193489/500000，尚未完成；最后一条 loss=0.0004106121，ETA 约 16 小时。
- CINN：未启动，未生成 CINN log。
- 后台任务 `bgl3ope38` 返回 exit code 1，但任务输出文件只包含启动信息；从 dy log 尾部看，训练过程本身持续正常输出到 step 193489，未看到模型侧 Python traceback。初步判断更像外层后台任务/会话超时或被终止，而不是 waveguide3D 训练本身报错。
