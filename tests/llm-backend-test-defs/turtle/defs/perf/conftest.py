import os
import time

import pytest
from trt_test.misc import call, check_call

from ..build_engines import *
from ..common import *


@pytest.fixture(scope="class")
def setup_gpt_python_backend_perf_test_env(tensorrt_llm_gpt_example_root,
                                           gpt_tokenizer_model_root):
    llm_backend_repo_root = os.environ["LLM_BACKEND_ROOT"]

    # Make sure Triton server are killed before each test.
    call(f"pkill -9 tritonserver", shell=True)
    time.sleep(2)

    # Build engine
    ENGINE_PATH = prepare_gpt_350m_engine(
        "python_backend",
        tensorrt_llm_gpt_example_root,
        gpt_tokenizer_model_root,
    )

    # Prepare model repo
    origin_model_repo = os.path.join(llm_backend_repo_root, "all_models",
                                     "gpt")
    new_model_repo = os.path.join(llm_backend_repo_root, "triton_repo")
    check_call(f"rm -rf {new_model_repo}", shell=True)
    check_call(f"cp -R {origin_model_repo} {new_model_repo}", shell=True)

    # Modify config.pbtxt
    TOKENIZER_PATH = gpt_tokenizer_model_root
    TOKENIZER_TYPE = "auto"
    fill_template_py = os.path.join(llm_backend_repo_root, "tools",
                                    "fill_template.py")
    llm_config = os.path.join(llm_backend_repo_root, "triton_repo",
                              "tensorrt_llm", "config.pbtxt")
    preprocessing_config = os.path.join(llm_backend_repo_root, "triton_repo",
                                        "preprocessing", "config.pbtxt")
    postprocessing_config = os.path.join(llm_backend_repo_root, "triton_repo",
                                         "postprocessing", "config.pbtxt")
    check_call(
        f"python3 {fill_template_py} -i {llm_config} engine_dir:{ENGINE_PATH}",
        shell=True)
    check_call(
        f"python3 {fill_template_py} -i {preprocessing_config} tokenizer_dir:{TOKENIZER_PATH},tokenizer_type:{TOKENIZER_TYPE}",
        shell=True)
    check_call(
        f"python3 {fill_template_py} -i {postprocessing_config} tokenizer_dir:{TOKENIZER_PATH},tokenizer_type:{TOKENIZER_TYPE}",
        shell=True)
    # Launch Triton Server
    launch_server_py = os.path.join(llm_backend_repo_root, "scripts",
                                    "launch_triton_server.py")
    check_call(
        f"python3 {launch_server_py} --world_size=1 --model_repo={new_model_repo}",
        shell=True)
    check_server_ready()
    yield
    # Gracefully terminate Triton Server after each test.
    call(f"pkill tritonserver", shell=True)
    time.sleep(8)


@pytest.fixture(scope="class")
def setup_llama_ifb_perf_test_env(tensorrt_llm_llama_example_root,
                                  llama_v2_tokenizer_model_root):
    llm_backend_repo_root = os.environ["LLM_BACKEND_ROOT"]
    DECOUPLED_MODE = "False"
    MAX_TOKENS_IN_KV_CACHE = ""
    MAX_KV_CACHE_LEN = ""
    BATCH_SCHEDULER_POLICY = "guaranteed_no_evict"
    BATCHING_STRATEGY = "inflight_fused_batching"
    MAX_NUM_SEQUENCE = ""
    KV_CACHE_FREE_GPU_MEM_FRACTION = ""
    EXCLUDE_INPUT_IN_OUTPUT = "False"
    ENABLE_TRT_OVERLAP = "False"
    TRITON_MAX_BATCH_SIZE = "128"
    MAX_QUEUE_DELAY_MICROSECONDS = "0"
    MAX_BEAM_WIDTH = "1"
    PREPROCESSING_INSTANCE_COUNT = "1"
    POSTPROCESSING_INSTANCE_COUNT = "1"
    ACCUMULATE_TOKEN = "False"
    BLS_INSTANCE_COUNT = "1"

    # Make sure Triton server are killed before each test.
    call(f"pkill -9 tritonserver", shell=True)
    time.sleep(2)

    # Build engine
    ENGINE_PATH = prepare_llama_v2_7b_engine("ifb",
                                             tensorrt_llm_llama_example_root,
                                             llama_v2_tokenizer_model_root)

    # Prepare model repo
    new_model_repo = os.path.join(llm_backend_repo_root, "triton_repo")
    prepare_ib_model_repo(llm_backend_repo_root, new_model_repo)

    # Modify config.pbtxt
    TOKENIZER_PATH = llama_v2_tokenizer_model_root
    TOKENIZER_TYPE = "llama"
    modify_ib_config_pbtxt(ENGINE_PATH, TOKENIZER_PATH, TOKENIZER_TYPE,
                           llm_backend_repo_root, DECOUPLED_MODE,
                           MAX_TOKENS_IN_KV_CACHE, MAX_KV_CACHE_LEN,
                           BATCH_SCHEDULER_POLICY, BATCHING_STRATEGY,
                           MAX_NUM_SEQUENCE, KV_CACHE_FREE_GPU_MEM_FRACTION,
                           EXCLUDE_INPUT_IN_OUTPUT, ENABLE_TRT_OVERLAP,
                           TRITON_MAX_BATCH_SIZE, MAX_QUEUE_DELAY_MICROSECONDS,
                           MAX_BEAM_WIDTH, PREPROCESSING_INSTANCE_COUNT,
                           POSTPROCESSING_INSTANCE_COUNT, ACCUMULATE_TOKEN,
                           BLS_INSTANCE_COUNT)

    # Launch Triton Server
    launch_server_py = os.path.join(llm_backend_repo_root, "scripts",
                                    "launch_triton_server.py")
    check_call(
        f"python3 {launch_server_py} --world_size=1 --model_repo={new_model_repo}",
        shell=True)
    check_server_ready()
    yield
    # Gracefully terminate Triton Server after each test.
    call(f"pkill tritonserver", shell=True)
    time.sleep(8)
