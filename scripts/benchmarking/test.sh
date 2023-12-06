#!/usr/bin/bash

MODEL=$1
ENGINE_PATH=$2
TOKENIZER_PATH=$3
TOKENIZER_TYPE=$4
BS=$5
MAX_INPUT_SEQLEN=$6
WORLD_SIZE=$7
RECORD_LOG=$8
MAX_ATTENTION_WINDOW_SIZE=$9

set -e
nvidia-smi

pushd ../../
source tools/utils.sh

#-----------------------  WORKLOAD_DETAILS -----------------------#

# token normal distribution.
# (ip_mean, ip_stdev, op_mean, op_stdev, num_prompts)
TOKEN_DIST_LIST=( "128,0,1,0,8192" "32,0,1024,0,1024" )
DATASETS=( "<dataset>" ) # names of datasets

# key: dataset name, value: path to dataset json file
declare -A dataset_dict=( ["<dataset>"]="<dataset_path>" )

# dictionary[workload] =  list of request rates to shmoo over. Should contain keys from TOKEN_DIST_LIST and DATASETS
declare -A REQ_RATES=(  ["128,0,1,0,8192"]="-1"
                        ["32,0,1024,0,1024"]="-1"
                        ["cnn"]="-1"
                        ["openweb"]="-1"
                    )
REQ_RATES_HIST="" #
#-----------------------------------------------------------------#

EXCLUDE_INPUT_IN_OUTPUT="false"
ENABLE_TRT_OVERLAP="false"
MAX_QUEUE_DELAY_MICROSECONDS="0"
MAX_BEAM_WIDTH="1"

gpu_info=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)
if [[ $gpu_info == *"H100"* ]]; then
    MACHINE="H100"
elif [[ $gpu_info == *"A100"* ]]; then
    MACHINE="A100"
elif [[ $gpu_info == *"L40S"* ]]; then
    MACHINE="L40S"
fi

fill_triton_repo () {
    # Modify config.pbtxt
    python3 tools/fill_template.py -i my_models/inflight_batcher_llm/tensorrt_llm/config.pbtxt engine_dir:${ENGINE_PATH},decoupled_mode:"False",batching_strategy:${BATCHING_STRATEGY},max_attention_window_size:${MAX_ATTENTION_WINDOW_SIZE},batch_scheduler_policy:${BATCH_SCHEDULER_POLICY},exclude_input_in_output:${EXCLUDE_INPUT_IN_OUTPUT},triton_max_batch_size:${BS},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MICROSECONDS},max_beam_width:${MAX_BEAM_WIDTH},enable_trt_overlap:${ENABLE_TRT_OVERLAP}
    python3 tools/fill_template.py -i my_models/inflight_batcher_llm/preprocessing/config.pbtxt triton_max_batch_size:${BS},tokenizer_dir:${TOKENIZER_PATH},tokenizer_type:${TOKENIZER_TYPE}
    python3 tools/fill_template.py -i my_models/inflight_batcher_llm/postprocessing/config.pbtxt triton_max_batch_size:${BS},tokenizer_dir:${TOKENIZER_PATH},tokenizer_type:${TOKENIZER_TYPE}
    python3 tools/fill_template.py -i my_models/inflight_batcher_llm/ensemble/config.pbtxt triton_max_batch_size:${BS}

}

print_test_params () {

    echo "----------------------------------"
    echo " Test parameters:"
    echo "----------------------------------"
    echo "BATCHING_STRATEGY: ${BATCHING_STRATEGY}"
    echo "BATCH_SCHEDULER_POLICY: ${BATCH_SCHEDULER_POLICY}"
    echo "ENABLE_TRT_OVERLAP: ${ENABLE_TRT_OVERLAP}"
    echo "EXCLUDE_INPUT_IN_OUTPUT: ${EXCLUDE_INPUT_IN_OUTPUT}"
    echo "TRITON_MAX_BATCH_SIZE: ${BS}"
    echo "MAX_QUEUE_DELAY_MICROSECONDS: ${MAX_QUEUE_DELAY_MICROSECONDS}"
    echo "MAX_BEAM_WIDTH: ${MAX_BEAM_WIDTH}"
    echo "MAX_ATTENTION_WINDOW_SIZE: ${MAX_ATTENTION_WINDOW_SIZE}"
    echo "----------------------------------"
}

if true; then
    echo "TRUE"

    BATCHING_STRATEGIES=( "inflight_fused_batching" "v1" )

    for BATCHING_STRATEGY in "${BATCHING_STRATEGIES[@]}"; do

        BATCH_SCHEDULER_POLICIES=( "guaranteed_no_evict" )

        for BATCH_SCHEDULER_POLICY in "${BATCH_SCHEDULER_POLICIES[@]}"; do

            echo -e " \n ================= INITIALIZING TRITONSERVER FOR =============== \n"
            print_test_params

            # Start each server with fresh configuration
            rm -rf my_models
            cp -R all_models my_models

            fill_triton_repo

            if [ "$RECORD_LOG" == "true" ]; then
                echo -e " \n ========= Collecting log for the server ======== \n"
                python3 scripts/launch_triton_server.py --world_size $WORLD_SIZE --model_repo my_models/inflight_batcher_llm/ --log --log-file triton_log.txt
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

                for DATASET in "${DATASETS[@]}"; do
                    IFS=',' read -ra REQUEST_RATES <<< "${REQ_RATES[${DATASET}]}"
                    for REQ_RATE in "${REQUEST_RATES[@]}"; do
                        op_stats_name="${MACHINE}__${MODEL}__${BATCHING_STRATEGY}__${BATCH_SCHEDULER_POLICY}__${DATASET}__${REQ_RATE}"
                        op_stats_csv_name="$op_stats_name.csv"

                        echo -e "DATASET: $DATASET \n\n"
                        echo -e " ======== BENCHMARK_CORE_MODEL --> OP STATS FILE = ${op_stats_csv_name} ============== \n"
                        dataset_path="${dataset_dict[$DATASET]}"
                        python3 benchmark_core_model.py \
                            -i grpc --max-input-len $MAX_INPUT_SEQLEN \
                            --request-rate $REQ_RATE --op-stats-csv "$op_stats_csv_name" \
                            --num-requests 3000 \
                            dataset \
                            --dataset $dataset_path \
                            --tokenizer-dir "$TOKENIZER_PATH" --tokenizer-type "$TOKENIZER_TYPE"

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
                            echo -e " ======== BENCHMARK_CORE_MODEL --> OP STATS FILE = ${op_stats_csv_name} ============== \n"
                            python3 benchmark_core_model.py \
                                -i grpc --max-input-len $MAX_INPUT_SEQLEN \
                                --request-rate $REQ_RATE --op-stats-csv "$op_stats_csv_name" \
                                --num-requests $num_prompts \
                                token-norm-dist \
                                --input-mean $ip_mean --input-stdev $ip_stdev --output-mean $op_mean --output-stdev $op_stdev \


                            sleep 5
                    done
                done

                IFS=',' read -ra REQUEST_RATES <<< $REQ_RATES_HIST
                for REQ_RATE in "${REQUEST_RATES[@]}"; do

                        op_stats_name="${MACHINE}__${MODEL}__${BATCHING_STRATEGY}__${BATCH_SCHEDULER_POLICY}__token-hist-example__${REQ_RATE}"
                        op_stats_csv_name="$op_stats_name.csv"
                        echo -e "DATASET: token-hist-example \n\n"
                        echo -e " ======== BENCHMARK_CORE_MODEL --> OP STATS FILE = ${op_stats_csv_name} ============== \n"
                        python3 benchmark_core_model.py \
                            -i grpc --max-input-len $MAX_INPUT_SEQLEN \
                            --request-rate $REQ_RATE --op-stats-csv "$op_stats_csv_name" \
                            token-from-histogram --histogram-key example

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
