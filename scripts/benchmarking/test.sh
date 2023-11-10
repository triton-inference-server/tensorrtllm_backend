#!/usr/bin/bash

MODEL=$1
ENGINE_PATH=$2
TOKENIZER_PATH=$3
TOKENIZER_TYPE=$4
BS=$5
WORLD_SIZE=$6
RECORD_LOG=$7

set -e
nvidia-smi

pushd ../../
source tools/utils.sh

#-----------------------  WORKLOAD_DETAILS -----------------------#

# token normal dist.
# (ip_mean, ip_stdev, op_mean, op_stdev, num_prompts)
TOKEN_DIST_LIST=( "128,10,20,2,500" "20,2,128,10,500" )
DATASETS=( "cnn" ) # names of datasets

# key: dataset name, value: path to dataset json file
declare -A dataset_dict=( []"cnn"]="<dataset_path>" )

# dictionary[workload] =  list of request rates . Should contain keys from TOKEN_DIST_LIST and DATASETS
declare -A REQ_RATES=(  ["128,10,20,2,500"]="-1,20"
                        ["20,2,128,10,500"]="-1,40"
                        ["cnn"]="-1,60"
                    )
#-----------------------------------------------------------------#


if true; then
    echo "TRUE"

    BATCHING_STRATEGIES=( "inflight_fused_batching" ) #"v1"


    for BATCHING_STRATEGY in "${BATCHING_STRATEGIES[@]}"; do

        if [ "$BATCHING_STRATEGY" = "inflight_fused_batching" ]; then
            BATCH_SCHEDULER_POLICIES=( "guaranteed_no_evict" )
        else
            BATCH_SCHEDULER_POLICIES=( "max_utilization" )
        fi

        for BATCH_SCHEDULER_POLICY in "${BATCH_SCHEDULER_POLICIES[@]}"; do

            echo -e " \n ================= INITIALIZING TRITONSERVER FOR =============== \n"
            echo "BATCH_SCHEDULER_POLICY: ${BATCH_SCHEDULER_POLICY}"
            echo "BATCHING SCHEME: ${BATCHING_STRATEGY}"

            # Start each server with fresh configuration
            rm -rf my_models
            cp -R all_models my_models

            # Modify config.pbtxt
            python3 tools/fill_template.py -i my_models/inflight_batcher_llm/tensorrt_llm/config.pbtxt engine_dir:${ENGINE_PATH},decoupled_mode:"False",batching_strategy:${BATCHING_STRATEGY},batch_scheduler_policy:${BATCH_SCHEDULER_POLICY},exclude_input_in_output:"false",triton_max_batch_size:${BS},max_queue_delay_microseconds:"0",max_batch_width:"1"
            python3 tools/fill_template.py -i my_models/inflight_batcher_llm/preprocessing/config.pbtxt triton_max_batch_size:${BS},tokenizer_dir:${TOKENIZER_PATH},tokenizer_type:${TOKENIZER_TYPE}
            python3 tools/fill_template.py -i my_models/inflight_batcher_llm/postprocessing/config.pbtxt triton_max_batch_size:${BS},tokenizer_dir:${TOKENIZER_PATH},tokenizer_type:${TOKENIZER_TYPE}

            if [ "$RECORD_LOG" == "true" ]; then
                echo -e " \n ========= Collecting log for the server ======== \n"
                python3 scripts/launch_triton_server.py --world_size $WORLD_SIZE --model_repo my_models/inflight_batcher_llm/ --tritonserver "/opt/tritonserver/bin/tritonserver --log-verbose 3 --log-file triton_log.txt"
            else
                python3 scripts/launch_triton_server.py --world_size $WORLD_SIZE --model_repo my_models/inflight_batcher_llm/
            fi
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
                    MAX_INPUT_LEN=128000
                fi

                for DATASET in "${DATASETS[@]}"; do
                    IFS=',' read -ra REQUEST_RATES <<< "${REQ_RATES[${DATASET}]}"
                    for REQ_RATE in "${REQUEST_RATES[@]}"; do
                        op_stats_name="${MACHINE}__${MODEL}__${BATCHING_STRATEGY}__${BATCH_SCHEDULER_POLICY}__${DATASET}__${REQ_RATE}"
                        op_stats_csv_name="$op_stats_name.csv"

                        echo -e "DATASET: $DATASET \n\n"
                        echo -e " ======== IDENTITY_TEST --> OP STATS FILE = ${op_stats_csv_name} ============== \n"
                        dataset_path="${dataset_dict[$DATASET]}"
                        # Identity test
                        python3 identity_test.py \
                            -i grpc --max_input_len $MAX_INPUT_LEN \
                            --request_rate $REQ_RATE --op_stats_csv "$op_stats_csv_name" \
                            --num_requests 3000 \
                            dataset \
                            --dataset $dataset_path \
                            --tokenizer_dir "$TOKENIZER_PATH" --tokenizer_type "$TOKENIZER_TYPE"

                        sleep 5
                    done
                done

                for TOKEN_DIST in "${TOKEN_DIST_LIST[@]}"; do
                    IFS=',' read -ra REQUEST_RATES <<< "${REQ_RATES[${TOKEN_DIST}]}"
                    for REQ_RATE in "${REQUEST_RATES[@]}"; do

                            # Use IFS and read to split the string into an array
                            IFS=',' read -ra token_params <<< "$TOKEN_DIST"
                            ip_mean=${token_params[0]}
                            ip_stdev=${token_params[1]}
                            op_mean=${token_params[2]}
                            op_stdev=${token_params[3]}
                            num_prompts=${token_params[4]}

                            op_stats_name="${MACHINE}__${MODEL}__${BATCHING_STRATEGY}__${BATCH_SCHEDULER_POLICY}__normal-token-dist-${ip_mean}-${ip_stdev}-${op_mean}-${op_stdev}__${REQ_RATE}"
                            op_stats_csv_name="$op_stats_name.csv"
                            echo -e "DATASET: normal-token-dist \n\n"
                            echo -e " ======== IDENTITY_TEST --> OP STATS FILE = ${op_stats_csv_name} ============== \n"
                            python3 identity_test.py \
                                -i grpc --max_input_len $MAX_INPUT_LEN \
                                --request_rate $REQ_RATE --op_stats_csv "$op_stats_csv_name" \
                                --num_requests $num_prompts \
                                token_norm_dist \
                                --input_mean $ip_mean --input_stdev $ip_stdev --output_mean $op_mean --output_stdev $op_stdev \


                            sleep 5
                    done
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
