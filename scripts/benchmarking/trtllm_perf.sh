#!/usr/bin/bash

MODEL=$1
RECORD_SERVER_STATS="${2:-"false"}"

TOKENIZER_DIR=/trt_llm_data/llm-models/llama-models/llama-7b-hf
TOKENIZER_TYPE=llama

GPT2=/trt_llm_data/llm-models/gpt2
OPT_125M=/trt_llm_data/llm-models/opt-125m
LLAMA=/trt_llm_data/llm-models/llama-models/llama-7b-hf
GPTJ=/trt_llm_data/llm-models/gpt-j-6b

set -e

########################   STATIC VALUES #######################

gpu_info=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)
script_dir=$(dirname "$(realpath "$0")")

declare -A bs_dict
if [[ $gpu_info == *"A100"* ]] ||  [[ $gpu_info == *"H100"* ]]; then
    bs_dict["llama-7b-fp8"]=2048
    bs_dict["llama-13b-fp8"]=1024
    bs_dict["gptj-6b-fp8"]=96
    bs_dict["llama-70b-fp8-tp2"]=64
    bs_dict["llama-70b-fp8-tp4"]=128
    bs_dict["falcon-180b-fp8-tp8"]=64
elif [[ $gpu_info == *"L40S"* ]]; then
    bs_dict["llama-7b-fp8"]=1024
    bs_dict["llama-13b-fp8"]=512
    bs_dict["gptj-6b-fp8"]=48
fi

if [ -z "$MODEL" ]; then
    echo "No model specified. Will run default list for the machine"
    if [[ $gpu_info == *"A100"* ]]; then
        model_list=(  "llama-7b-fp16" "gptj-6b-fp16" "llama-70b-fp16-tp4"  "falcon-180b-fp16-tp8" )
        machine="a100"
    elif [[ $gpu_info == *"H100"* ]]; then
        model_list=(  "llama-7b-fp8" "llama-70b-fp8-tp4"  "gptj-6b-fp8" "llama-70b-fp8-tp2" "falcon-180b-fp8-tp8" )
        machine="h100"
    elif [[ $gpu_info == *"L40S"* ]]; then
        model_list=(  "llama-7b-fp8" "gptj-6b-fp8"  )
        machine="l40s"
    else
        echo -e "Nothing to run for this machine"
    fi
else
    model_list=( "$MODEL" )
    machine="h100"
fi

for MODEL in "${model_list[@]}"; do

    BS="${bs_dict[$MODEL]}"
    isl=2048
    osl=512
    if [[ $MODEL == *"gptj-6b"* ]]; then
        isl=1535
    fi
    dir_name="bs${BS}_io_${isl}_${osl}"

    if [ "$MODEL" = "llama-7b-fp8" ]; then

        echo -e " \n ********  BUILDING $MODEL ************* \n"
        bash build_model.sh llama-7b-fp8 ${script_dir}/../../tensorrt_llm/trt_engines/${machine}/${MODEL}/${dir_name} $BS

        echo -e " \n ******** RUNNING $MODEL *************** \n"
        bash test.sh llama-7b-fp8 ${script_dir}/../../tensorrt_llm/trt_engines/${machine}/${MODEL}/${dir_name} $TOKENIZER_DIR $TOKENIZER_TYPE $BS 1 $RECORD_SERVER_STATS

    fi

    if [ "$MODEL" = "llama-13b-fp8" ]; then

        echo -e " \n ********  BUILDING $MODEL ************* \n"
        bash build_model.sh llama-13b-fp8 ${script_dir}/../../tensorrt_llm/trt_engines/${machine}/${MODEL}/${dir_name} $BS

        echo -e " \n ******** RUNNING $MODEL *************** \n"
        bash test.sh llama-13b-fp8 ${script_dir}/../../tensorrt_llm/trt_engines/${machine}/${MODEL}/${dir_name} $TOKENIZER_DIR $TOKENIZER_TYPE $BS 1 $RECORD_SERVER_STATS

    fi

    if [ "$MODEL" = "llama-7b-fp16" ]; then

        echo -e " \n ********  BUILDING $MODEL ************* \n"
        bash build_model.sh llama-7b-fp16 ${script_dir}/../../tensorrt_llm/trt_engines/${machine}/${MODEL}/${dir_name} $BS

        echo -e " \n ******** RUNNING $MODEL *************** \n"
        bash test.sh llama-7b-fp16 ${script_dir}/../../tensorrt_llm/trt_engines/${machine}/${MODEL}/${dir_name} $TOKENIZER_DIR $TOKENIZER_TYPE $BS 1 $RECORD_SERVER_STATS

    fi

    if [ "$MODEL" = "llama-70b-fp8-tp2" ]; then

        echo -e " \n ********  BUILDING $MODEL ************* \n"
        bash build_model.sh llama-70b-fp8-tp2 ${script_dir}/../../tensorrt_llm/trt_engines/${machine}/${MODEL}/${dir_name} $BS

        echo -e " \n ******** RUNNING $MODEL *************** \n"
        bash test.sh llama-70b-fp8-tp2 ${script_dir}/../../tensorrt_llm/trt_engines/${machine}/${MODEL}/${dir_name} $TOKENIZER_DIR $TOKENIZER_TYPE $BS 2 $RECORD_SERVER_STATS

    fi

    if [ "$MODEL" = "llama-70b-fp8-tp4" ]; then

        echo -e " \n ********  BUILDING $MODEL ************* \n"
        bash build_model.sh llama-70b-fp8-tp4 ${script_dir}/../../tensorrt_llm/trt_engines/${machine}/${MODEL}/${dir_name} $BS

        echo -e " \n ******** RUNNING $MODEL *************** \n"
        bash test.sh llama-70b-fp8-tp4 ${script_dir}/../../tensorrt_llm/trt_engines/${machine}/${MODEL}/${dir_name} $TOKENIZER_DIR $TOKENIZER_TYPE $BS 4 $RECORD_SERVER_STATS

    fi

     if [ "$MODEL" = "llama-70b-fp16-tp4" ]; then

        echo -e " \n ********  BUILDING $MODEL ************* \n"
        bash build_model.sh llama-70b-fp16-tp4 ${script_dir}/../../tensorrt_llm/trt_engines/${machine}/${MODEL}/${dir_name} $BS

        echo -e " \n ******** RUNNING $MODEL *************** \n"
        bash test.sh llama-70b-fp16-tp4 ${script_dir}/../../tensorrt_llm/trt_engines/${machine}/${MODEL}/${dir_name} $TOKENIZER_DIR $TOKENIZER_TYPE $BS 4 $RECORD_SERVER_STATS

    fi

    if [ "$MODEL" = "gptj-6b-fp8" ]; then

        echo -e " \n ********  BUILDING $MODEL ************* \n"
        bash build_model.sh gptj-6b-fp8 ${script_dir}/../../tensorrt_llm/trt_engines/${machine}/${MODEL}/${dir_name} $BS

        echo -e " \n ******** RUNNING $MODEL *************** \n"
        bash test.sh gptj-6b-fp8 ${script_dir}/../../tensorrt_llm/trt_engines/${machine}/${MODEL}/${dir_name} $TOKENIZER_DIR $TOKENIZER_TYPE $BS 1 $RECORD_SERVER_STATS

    fi

    if [ "$MODEL" = "gptj-6b-fp16" ]; then

        echo -e " \n ********  BUILDING $MODEL ************* \n"
        bash build_model.sh gptj-6b-fp16 ${script_dir}/../../tensorrt_llm/trt_engines/${machine}/${MODEL}/${dir_name} $BS

        echo -e " \n ******** RUNNING $MODEL *************** \n"
        bash test.sh gptj-6b-fp16 ${script_dir}/../../tensorrt_llm/trt_engines/${machine}/${MODEL}/${dir_name} $TOKENIZER_DIR $TOKENIZER_TYPE $BS 1 $RECORD_SERVER_STATS

    fi

    if [ "$MODEL" = "falcon-180b-fp8-tp8" ]; then

        echo -e " \n ********  BUILDING $MODEL ************* \n"
        bash build_model.sh falcon-180b-fp8-tp8 ${script_dir}/../../tensorrt_llm/trt_engines/${machine}/${MODEL}/${dir_name} $BS

        echo -e " \n ******** RUNNING $MODEL *************** \n"
        bash test.sh falcon-180b-fp8-tp8 ${script_dir}/../../tensorrt_llm/trt_engines/${machine}/${MODEL}/${dir_name} $TOKENIZER_DIR $TOKENIZER_TYPE $BS 8 $RECORD_SERVER_STATS

    fi

     if [ "$MODEL" = "falcon-180b-fp16-tp8" ]; then

        echo -e " \n ********  BUILDING $MODEL ************* \n"
        bash build_model.sh falcon-180b-fp16-tp8 ${script_dir}/../../tensorrt_llm/trt_engines/${machine}/${MODEL}/${dir_name} $BS

        echo -e " \n ******** RUNNING $MODEL *************** \n"
        bash test.sh falcon-180b-fp16-tp8 ${script_dir}/../../tensorrt_llm/trt_engines/${machine}/${MODEL}/${dir_name} $TOKENIZER_DIR $TOKENIZER_TYPE $BS 8 $RECORD_SERVER_STATS

    fi

done
