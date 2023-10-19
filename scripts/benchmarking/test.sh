#!/usr/bin/bash

MODEL=$1
ENGINE_PATH=$2
TOKENIZER_PATH=$3
TOKENIZER_TYPE=$4
BS=$5
WORLD_SIZE=$6

set -e
nvidia-smi
source ../tools/utils.sh

pushd ../../

declare -A dataset_dict
# key: dataset name, value: path to dataset json file
dataset_dict["<dataset_name>"]="<dataset_path>"

if true; then
    echo "TRUE"

    BATCH_ALGOS=( "v1" "inflight_fused_batching" )
    DATASETS=( ) # names of keys in dataset_dict to iterate over

    for BATCH_ALGO in "${BATCH_ALGOS[@]}"; do

        if [ "$BATCH_ALGO" = "inflight_fused_batching" ]; then
            BATCH_SCHEDULER_POLICIES=( "guaranteed_completion" )
        else
            BATCH_SCHEDULER_POLICIES=( "max_utilization" )
        fi

        for BATCH_SCHEDULER_POLICY in "${BATCH_SCHEDULER_POLICIES[@]}"; do

            echo -e " \n ================= INITIALIZING TRITONSERVER FOR =============== \n"
            echo "BATCH_SCHEDULER_POLICY: ${BATCH_SCHEDULER_POLICY}"
            echo "BATCHING SCHEME: ${BATCH_ALGO}"

            # Start each server with fresh configuration
            rm -rf my_models
            cp -R all_models my_models

            # Modify config.pbtxt
            python3 tools/fill_template.py -i my_models/inflight_batcher_llm/tensorrt_llm/config.pbtxt engine_dir:${ENGINE_PATH},decoupled_mode:False,batching_strategy:${BATCH_ALGO},batch_scheduler_policy:${BATCH_SCHEDULER_POLICY},enable_trt_overlap:True
            python3 tools/fill_template.py -i my_models/inflight_batcher_llm/preprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_PATH},tokenizer_type:${TOKENIZER_TYPE}
            python3 tools/fill_template.py -i my_models/inflight_batcher_llm/postprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_PATH},tokenizer_type:${TOKENIZER_TYPE}
            python3 scripts/benchmarking/replace_bs.py my_models/inflight_batcher_llm/ ${BS}

            python3 scripts/launch_triton_server.py --world_size $WORLD_SIZE --model_repo my_models/inflight_batcher_llm/
            # Use pgrep to find the PID of the "mpirun" process
            pid=$(pgrep mpirun)

            if [ -n "$pid" ]; then
                echo "PID of mpirun process: $pid"
            else
                echo "No mpirun process found."
            fi
            export SERVER_PID=($pid)
            wait_for_server_ready ${SERVER_PID} 1200

            pushd tools/inflight_batcher_llm/
            if [ $? -eq 0 ]; then

                if [[ "$MODEL" == *"fp8"* ]]; then
                    MACHINE="h100"
                else
                    MACHINE="a100"
                fi

                if [[ "$MODEL" == *"gptj"* ]]; then
                    MAX_INPUT_LEN=1535
                else
                    MAX_INPUT_LEN=2048
                fi

                for DATASET in "${DATASETS[@]}"; do
                    op_stats_csv_name="${MACHINE}__${MODEL}__${BATCH_ALGO}__${BATCH_SCHEDULER_POLICY}__${DATASET}.csv"
                    echo -e "DATASET: $DATASET \n\n"
                    echo -e " ======== IDENTITY_TEST --> OP STATS FILE = ${op_stats_csv_name} ============== \n"
                    dataset_path="${dataset_dict[$DATASET]}"
                    # Identity test
                    python3 identity_test.py \
                        --dataset $dataset_path \
                        -i grpc --max_input_len $MAX_INPUT_LEN \
                        --tokenizer_dir "$TOKENIZER_PATH" --tokenizer_type "$TOKENIZER_TYPE" \
                        --op_stats_csv "$op_stats_csv_name" --time_bet_reqs 0
                    sleep 5

                done

                echo -e " \n ========= KILLING TRITON SERVER WITH PID:  #$SERVER_PID  ============== \n"
                kill  -9 ${SERVER_PID}
            else
                echo -e "\n !!!!!!!!!!!!  Triton Server initialization failed !!!!!!!!!!!!!!! \n"
            fi

            popd # tools/inflight_batcher_llm
    done
    done
fi
