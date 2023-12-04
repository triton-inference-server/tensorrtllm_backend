#!/usr/bin/bash

# MODEL_SPEC is defined as "MODEL_NAME,TP,PP"
MODEL_SPEC=$1
RECORD_SERVER_STATS="${2:-"false"}"

TOKENIZER_DIR=/trt_llm_data/llm-models/llama-models/llama-7b-hf
TOKENIZER_TYPE=llama

set -e

########################   STATIC VALUES #######################

gpu_info=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)
script_dir=$(dirname "$(realpath "$0")")

declare -A bs_dict
if [[ $gpu_info == *"A100"* ]] ||  [[ $gpu_info == *"H100"* ]]; then
    bs_dict["llama-7b-fp8,1,1"]=2048
    bs_dict["llama-13b-fp8,1,1"]=2048
    bs_dict["llama-7b-fp16,1,1"]=1024
    bs_dict["llama-13b-fp16,1,1"]=1024
    bs_dict["gptj-6b-fp8,1,1"]=96
    bs_dict["llama-70b-fp8,2,1"]=512
    bs_dict["llama-70b-fp8,4,1"]=1024
    bs_dict["llama-70b-fp16,2,1"]=256
    bs_dict["llama-70b-fp16,4,1"]=512
    bs_dict["falcon-180b-fp8,8,1"]=512
elif [[ $gpu_info == *"L40S"* ]]; then
    bs_dict["llama-7b-fp8,1,1"]=1024
    bs_dict["llama-13b-fp8,1,1"]=512
    bs_dict["gptj-6b-fp8,1,1"]=1024
    bs_dict["llama-70b-fp8,2,1"]=256
    bs_dict["llama-70b-fp8,4,1"]=256
    bs_dict["llama-70b-fp8,1,4"]=256
    bs_dict["llama-70b-fp16,4,1"]=256
    bs_dict["llama-70b-fp16,1,4"]=256
fi

MAX_TOKENS=50000

if [ -z "$MODEL_SPEC" ]; then
    echo "No model spec specified. Will run default list for the MACHINE"

    if [[ $gpu_info == *"A100"* ]]; then
        MODEL_SPEC_LIST=( "llama-7b-fp16,1,1" "llama-13b-fp16,1,1"  "gptj-6b-fp16,1,1" "llama-70b-fp16,4,1"  "falcon-180b-fp16,8,1" )
        MACHINE="a100"
    elif [[ $gpu_info == *"H100"* ]]; then
        MODEL_SPEC_LIST=( "llama-7b-fp8,1,1" "llama-13b-fp8,1,1" "llama-70b-fp8,4,1" "gptj-6b-fp8,1,1" "llama-70b-fp8,2,1" "falcon-180b-fp8,8,1" )
        MACHINE="h100"
    elif [[ $gpu_info == *"L40S"* ]]; then
        MODEL_SPEC_LIST=( "llama-7b-fp8,1,1" "llama-13b-fp8,1,1" "gptj-6b-fp8,1,1" "llama-70b-fp8,2,1" "llama-70b-fp8,4,1" "llama-70b-fp8,1,4" )
        MACHINE="l40s"
    else
        echo -e "Nothing to run for this MACHINE"
    fi
else
    MODEL_SPEC_LIST=( "$MODEL_SPEC" )
    MACHINE="h100"
fi

for MODEL_SPEC in "${MODEL_SPEC_LIST[@]}"; do
    IFS=',' read -ra MODEL_SPECS <<< "${MODEL_SPEC}"
    MODEL=${MODEL_SPECS[0]}
    TP=${MODEL_SPECS[1]}
    PP=${MODEL_SPECS[2]}
    WORLD_SIZE=$((TP*PP))

    BS=${bs_dict[${MODEL_SPEC}]}
    MAX_INPUT_SEQLEN=16384
    MAX_OUTPUT_SEQLEN=4096
    if [[ $MODEL == *"gptj-6b"* ]]; then
        MAX_INPUT_SEQLEN=1535
        MAX_OUTPUT_SEQLEN=512
    fi
    DIR="bs${BS}_tokens${MAX_TOKENS}_tp${TP}_pp${PP}_isl${MAX_INPUT_SEQLEN}_osl${MAX_OUTPUT_SEQLEN}"
    ENGINE_PATH=${script_dir}/../../tensorrt_llm/trt_engines/${MACHINE}/${MODEL}/${DIR}

    echo -e " \n ********  BUILDING $MODEL with TP=$TP PP=$PP  ************* \n"
    bash build_model.sh $MODEL $ENGINE_PATH $BS $MAX_INPUT_SEQLEN $MAX_OUTPUT_SEQLEN $MAX_TOKENS $TP $PP $WORLD_SIZE

    echo -e " \n ******** RUNNING $MODEL with TP=$TP PP=$PP  *************** \n"
    bash test.sh $MODEL $ENGINE_PATH $TOKENIZER_DIR $TOKENIZER_TYPE $BS $MAX_INPUT_SEQLEN $TP $PP $WORLD_SIZE $RECORD_SERVER_STATS

done
