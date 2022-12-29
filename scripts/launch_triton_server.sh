WORK_DIR="${PWD}"
TRITONSERVER="/opt/tritonserver/bin/tritonserver"
MODEL_REPO="${WORK_DIR}/all_models/"

export PYTHONPATH=${WORK_DIR}:${PYTHONPATH}

# num of processes = world_size = tp_size * pp_size
mpirun --allow-run-as-root \
    -n 1 ${TRITONSERVER} --model-repository=${MODEL_REPO} --backend-config=python,shm-region-prefix-name=prefix0_ : \
    -n 1 ${TRITONSERVER} --model-repository=${MODEL_REPO} --backend-config=python,shm-region-prefix-name=prefix1_ : \
    -n 1 ${TRITONSERVER} --model-repository=${MODEL_REPO} --backend-config=python,shm-region-prefix-name=prefix2_ : \
    -n 1 ${TRITONSERVER} --model-repository=${MODEL_REPO} --backend-config=python,shm-region-prefix-name=prefix3_ &
