model_item=cylinder-cylinder_2d
bs_item=1
fp_item=fp32
run_mode=DP
device_num=N1C1
pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple
# copy files
\cp test_tipc/dynamicTostatic/cylinder-cylinder_2d/benchmark_common/prepare.sh ./
\cp test_tipc/dynamicTostatic/cylinder-cylinder_2d/benchmark_common_infer/run_benchmark.sh ./
\cp test_tipc/dynamicTostatic/cylinder-cylinder_2d/benchmark_common/analysis_log.py ./
# prepare
bash prepare.sh
# run
to_static=True FLAGS_enable_pir_in_executor=true FLAGS_enable_pir_api=True FLAGS_cinn_bucket_compile=True FLAGS_group_schedule_tiling_first=1 FLAGS_cinn_new_group_scheduler=1 FLAGS_nvrtc_compile_to_cubin=True bash run_benchmark.sh ${model_item} ${bs_item} ${fp_item} ${run_mode} ${device_num} 2>&1;
sleep 10;
