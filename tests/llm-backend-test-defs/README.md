# TensorRT LLM Backend test definitions

The following subfolder contains test definitions for `TURTLE` (https://gitlab-master.nvidia.com/TensorRT/Infrastructure/turtle),
which are used to validate TensorRT LLM Backend.


## Directory structure

~~~
.
└── turtle              # TURTLE-related definitions
    ├── defs            #     Test definitions (pytest functions)
    └── test_lists      #     TURTLE-related test lists
        └── bloom       #         Test lists used by bloom automation
        └── qa          #         Test lists used by QA
~~~

## How to run turtle test locally for TRT-LLM-Backend?

1. Clone turtle lib，recommend to put it outside the TRT-LLM-Backend repo, to avoid nested git repo.

```bash
# Clone turtle to the same parent directory of TRT-LLM-Backend repo
git clone ssh://git@gitlab-master.nvidia.com:12051/TensorRT/Infrastructure/turtle.git
```

2. Example commands to run turtle test inside docker container
```bash
# Mount model weights data before launching docker container
mkdir ${PWD}/llm_data/
sudo mount 10.117.145.14:/vol/scratch1/scratch.michaeln_blossom ${PWD}/llm_data/

# Launch docker container
sudo docker run --gpus all --shm-size=2g --ulimit memlock=-1 --rm -it \
                    -v ${PWD}/llm_data/llm-models:/code/llm-models -v ${PWD}/tekit_backend:/code/tekit_backend \
                    -v ${PWD}/turtle:/code/turtle urm.nvidia.com/sw-tensorrt-docker/tensorrt-llm:dev-triton-23.10-trt9.2.0.5-2 bash

# In docker container
export LLM_MODELS_ROOT=/code/llm-models # turtle needs model weights to build engine
export LLM_BACKEND_ROOT=/code/tekit_backend/ # turtle test definition needs to read LLM_BACKEND_ROOT env to find where the example and unit tests code are

# Make sure tensorrt-llm python lib is installed properly

# Make sure libtriton_tensorrtllm.so is deployed properly

# Run through test list file
./turtle/bin/trt_test -D tekit_backend/tests/llm-backend-test-defs/turtle/defs/ \
                    -f tekit_backend/tests/llm-backend-test-defs/turtle/test_lists/qa/llm_backend_functional_overall.txt \
                    --test-python3-exe /usr/bin/python3 --output-dir output --save-workspace
# Run through test keyword
./turtle/bin/trt_test -D tekit_backend/tests/llm-backend-test-defs/turtle/defs/ \
                    -k test_gpt_350m \
                    --test-python3-exe /usr/bin/python3 --save-workspace

# List all available tests, by using "-l" option
./turtle/bin/trt_test -D tekit_backend/tests/llm-backend-test-defs/turtle/defs/ -l

# Run perf test
./turtle/bin/trt_test -D tekit_backend/tests/llm-backend-test-defs/turtle/defs/ \
                    -f llm_backend_perf_overall.txt \
                    --test-python3-exe /usr/bin/python3 --perf-log-formats csv \
                    --perf-clock-gpu-configs-file /code/tekit_backend/tests/llm-backend-test-defs/turtle/perf_configs/gpu_configs.yml \
                    --perf
