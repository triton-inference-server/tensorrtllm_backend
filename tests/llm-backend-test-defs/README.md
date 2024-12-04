# TensorRT LLM Backend test definitions

The following subfolder contains test definitions for `TURTLE` (https://gitlab-master.nvidia.com/TensorRT/Infrastructure/turtle),
which are used to validate TensorRT LLM Backend.


## Directory structure

~~~
.
└── turtle              # TURTLE-related definitions
    ├── defs            #     Test definitions (pytest functions)
    ├── perf_configs    #     Defines sm_clk and mem_clk used for perf testing
    └── test_lists      #     TURTLE-related test lists
        └── bloom       #         Test lists used by bloom automation
        └── qa          #         Test lists used by QA
~~~

## How to run turtle test locally for TRT-LLM-Backend?

### Take gpt-350m inflight batching test case as example
- Download turtle and tekit_backend and llm-qa-test
```shell
mkdir ~/workspace && cd ~/workspace
git clone ssh://git@gitlab-master.nvidia.com:12051/TensorRT/Infrastructure/turtle.git
git clone --recurse-submodules ssh://git@gitlab-master.nvidia.com:12051/ftp/tekit_backend.git
```

- Mount data server
```shell
mkdir -p ~/workspace/llm_data
sudo mount -o ro 10.117.145.14:/vol/scratch1/scratch.michaeln_blossom ~/workspace/llm_data/
```

- Launch docker container
```shell
sudo docker run --gpus all --shm-size=32g --ulimit memlock=-1 --rm -it -e LLM_MODELS_ROOT=/code/llm-models -v ${PWD}/llm_data/llm-models:/code/llm-models -v ${PWD}/tekit_backend:/code/tekit_backend -v ${PWD}/turtle:/code/turtle urm.nvidia.com/sw-tensorrt-docker/tensorrt-llm:tritonserver-24.10-py3-x86_64-ubuntu22.04-trt10.6.0.26-pypi-devel-202411041524-861 bash
```

- **In Container**
    - Set env
    ```shell
    export LLM_BACKEND_ROOT=/code/tekit_backend/
    export SKIP_CLEANUP_ENGINES=True
    ```
    - Build wheels and install
    ```shell
    cd /code/tekit_backend/tensorrt_llm
    python3 scripts/build_wheel.py --clean --trt_root /usr/local/tensorrt
    pip3 install build/tensorrt_llm-*.whl
    ```
    - Build IFB lib and deploy
    ```shell
    cd /code/tekit_backend/inflight_batcher_llm
    bash scripts/build.sh
    mkdir /opt/tritonserver/backends/tensorrtllm/
    cp build/libtriton_tensorrtllm.so /opt/tritonserver/backends/tensorrtllm/
    cp build/trtllmExecutorWorker /opt/tritonserver/backends/tensorrtllm/
    ```
    - Run TURTLE test.
    ```shell
    cd /code
    apt-get update && apt-get install -y libffi-dev
    # Run TURTLE with "-k" to match test name, e.g. "-k test_gpt_350m_ib" to test all the (sub-)test case contains "test_gpt_350m_ib" in test name.
    ./turtle/bin/trt_test -D tekit_backend/tests/llm-backend-test-defs/turtle/defs/ --test-python3-exe /usr/bin/python3 --save-workspace -k test_gpt_350m_ib

    # RUN TURTLE with "-f" to run a list of tests, e.g. run the L0 test list
    ./turtle/bin/trt_test -D tekit_backend/tests/llm-backend-test-defs/turtle/defs/ --test-python3-exe /usr/bin/python3 --save-workspace -f tekit_backend/tests/llm-backend-test-defs/turtle/test_lists/bloom/l0_functional.txt
    ```


### Tips
- To list all test available (In container)
```shell
cd /code
./turtle/bin/trt_test -D tekit_backend/tests/llm-backend-test-defs/turtle/defs/ -l
```
- To run perf test (In container)
```shell
cd /code
./turtle/bin/trt_test -D tekit_backend/tests/llm-backend-test-defs/turtle/defs/ \
                    --test-python3-exe /usr/bin/python3 --save-workspace \
                    --perf-log-formats csv \
                    --perf-clock-gpu-configs-file /code/tekit_backend/tests/llm-backend-test-defs/turtle/perf_configs/gpu_configs.yml \
                    --perf \
                    -k test_perf[gpt_350m-bs:1-input_output_len:128,8-num_runs:10]
