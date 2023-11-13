import os

import pytest
from trt_test.misc import check_call, print_info

from .common import *
from .conftest import venv_check_call


def get_rcca_path():
    cur_path = os.path.abspath(os.path.dirname(__file__))
    rcca_path = os.path.join(cur_path, "rcca")
    return rcca_path


@pytest.mark.parametrize("MAX_NUM_SEQUENCE", [""])
@pytest.mark.parametrize("MAX_TOKENS_IN_KV_CACHE", [""])
@pytest.mark.parametrize("BATCH_SCHEDULER_POLICY", ["guaranteed_no_evict"])
@pytest.mark.parametrize("KV_CACHE_FREE_GPU_MEM_FRACTION", [""])
@pytest.mark.parametrize("ENABLE_TRT_OVERLAP", ["False"],
                         ids=["disableTrtOverlap"])
@pytest.mark.parametrize("BATCHING_STRATEGY", ["V1"])
@pytest.mark.parametrize("DECOUPLED_MODE", ["False"],
                         ids=["disableDecoupleMode"])
@pytest.mark.parametrize("TRITON_MAX_BATCH_SIZE", ["128"])
@pytest.mark.parametrize("MAX_QUEUE_DELAY_MICROSECONDS", ["0"])
@pytest.mark.parametrize("MAX_BEAM_WIDTH", ["1"])
@pytest.mark.parametrize("EXCLUDE_INPUT_IN_OUTPUT", ["False"])
def test_rcca_bug_4323566(MAX_NUM_SEQUENCE, MAX_TOKENS_IN_KV_CACHE,
                          BATCH_SCHEDULER_POLICY,
                          KV_CACHE_FREE_GPU_MEM_FRACTION, ENABLE_TRT_OVERLAP,
                          BATCHING_STRATEGY, DECOUPLED_MODE,
                          TRITON_MAX_BATCH_SIZE, MAX_QUEUE_DELAY_MICROSECONDS,
                          MAX_BEAM_WIDTH, EXCLUDE_INPUT_IN_OUTPUT,
                          inflight_batcher_llm_client_root,
                          gpt_tokenizer_model_root, llm_backend_venv):
    if BATCHING_STRATEGY == "V1" and BATCH_SCHEDULER_POLICY == "max_utilization":
        print_info("Skipping. V1 doesn't support max_utilization.")
        return

    llm_backend_repo_root = os.environ["LLM_BACKEND_ROOT"]
    # Prepare model repo
    new_model_repo = os.path.join(llm_backend_repo_root, "triton_repo")
    prepare_ib_model_repo(llm_backend_repo_root, new_model_repo)

    # Modify config.pbtxt
    ENGINE_PATH = os.path.join(
        llm_backend_repo_root,
        "tensorrt_llm/examples/gpt/trt_engine/rcca-nvbug-4323566/")
    TOKENIZER_PATH = gpt_tokenizer_model_root
    TOKENIZER_TYPE = "auto"
    modify_ib_config_pbtxt(
        ENGINE_PATH, TOKENIZER_PATH, TOKENIZER_TYPE, llm_backend_repo_root,
        DECOUPLED_MODE, MAX_TOKENS_IN_KV_CACHE, BATCH_SCHEDULER_POLICY,
        BATCHING_STRATEGY, MAX_NUM_SEQUENCE, KV_CACHE_FREE_GPU_MEM_FRACTION,
        EXCLUDE_INPUT_IN_OUTPUT, ENABLE_TRT_OVERLAP, TRITON_MAX_BATCH_SIZE,
        MAX_QUEUE_DELAY_MICROSECONDS, MAX_BEAM_WIDTH)

    # Launch Triton Server
    launch_server_py = os.path.join(llm_backend_repo_root, "scripts",
                                    "launch_triton_server.py")
    check_call(
        f"python3 {launch_server_py} --force --world_size 1 --model_repo={new_model_repo}",
        shell=True)
    check_server_ready()
    # Run Test
    script_path = os.path.join(get_rcca_path(), "bug_4323566",
                               "inflight_batcher_llm_client_with_end_id.py")
    run_cmd = [
        f"{script_path}",
        f"--request-output-len=200",
    ]

    venv_check_call(llm_backend_venv, run_cmd)
