#!/usr/bin/env bash

# Test inference benchmark for a model (dynamic-to-static + CINN).
export LD_LIBRARY_PATH=/usr/lib64:${LD_LIBRARY_PATH}

function _set_params(){
    model_item=${1:-"model_item"}
    base_batch_size=${2:-"1"}
    fp_item=${3:-"fp32"}
    run_mode=${4:-"DP"}
    device_num=${5:-"N1C1"}

    backend="paddle"
    model_repo="modulus-sym"
    speed_unit="ms/iteration"
    skip_steps=0
    keyword="time/iteration:"
    convergence_key=""

    model_name=${model_item}_bs${base_batch_size}_${fp_item}_${run_mode}
    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=(${device})
    num_gpu_devices=${#arr[*]}
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}
    speed_log_path=${LOG_PATH_INDEX_DIR:-$(pwd)}
    train_log_file=${run_log_path}/${model_repo}_${model_name}_${device_num}_d2sT_infer_log
    speed_log_file=${speed_log_path}/${model_repo}_${model_name}_${device_num}_d2sT_infer_speed
}

function _analysis_log(){
    echo "infer_log_file: ${train_log_file}"
    echo "speed_log_file: ${speed_log_file}"
    cmd="python analysis_log.py --filename ${train_log_file}         --speed_log_file ${speed_log_file}         --model_name ${model_name}         --base_batch_size ${base_batch_size}         --run_mode ${run_mode}         --fp_item ${fp_item}         --keyword ${keyword}         --skip_steps ${skip_steps}         --device_num ${device_num} "
    echo ${cmd}
    eval $cmd
}

function _infer(){
    export NVIDIA_TF32_OVERRIDE=1
    echo "current CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, model_name=${model_name}, device_num=${device_num}, is profiling=${profiling}"

    export DDE_BACKEND=paddle
    export INFER_WARMUP=${INFER_WARMUP:-10}
    export INFER_STEPS=${INFER_STEPS:-100}
    export debug=${debug:-1}
    export loss_monitor=${loss_monitor:-1}
    infer_cmd="pushd examples/ldc; python ldc_2d_domain_decomposition_infer.py training.max_steps=0; popd"
    echo "pwd: $PWD infer_cmd: ${infer_cmd} log_file: ${train_log_file}"
    set -x
    timeout 60m bash -c "${infer_cmd}" > ${train_log_file} 2>&1
    if [ $? -ne 0 ];then
        echo -e "Generate ${model_name}, FAIL"
    else
        echo -e "Generate ${model_name}, SUCCESS"
    fi
}

_set_params $@
str_tmp=$(echo `pip list|grep paddlepaddle-gpu|awk -F ' ' '{print $2}'`)
export frame_version=${str_tmp%%.post*}
export frame_commit=$(echo `python -c "import paddle;print(paddle.version.commit)"`)
export model_branch=`git symbolic-ref HEAD 2>/dev/null | cut -d"/" -f 3`
export model_commit=$(git log|head -n1|awk '{print $2}')
echo "---------frame_version is ${frame_version}"
echo "---------Paddle commit is ${frame_commit}"
echo "---------Model commit is ${model_commit}"
echo "---------model_branch is ${model_branch}"

job_bt=`date '+%Y%m%d%H%M%S'`
_infer
job_et=`date '+%Y%m%d%H%M%S'`
export model_run_time=$((${job_et}-${job_bt}))
_analysis_log
