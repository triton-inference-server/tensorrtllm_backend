import os
import re

import pytest
from trt_test.misc import call, check_call, print_info

from .common import *
from .conftest import venv_check_call, venv_check_output


@pytest.fixture(autouse=True)
def stop_triton_server():
    # Make sure Triton server are killed before each test.
    call(f"pkill -9 tritonserver", shell=True)
    time.sleep(2)
    yield
    # Gracefully terminate Triton Server after each test.
    call(f"pkill tritonserver", shell=True)
    time.sleep(8)


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
        f"python3 {launch_server_py} --world_size=1 --model_repo={new_model_repo}",
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


@pytest.mark.parametrize("MAX_NUM_SEQUENCE", [""])
@pytest.mark.parametrize("MAX_TOKENS_IN_KV_CACHE", [""])
@pytest.mark.parametrize("BATCH_SCHEDULER_POLICY", ["guaranteed_no_evict"])
@pytest.mark.parametrize("KV_CACHE_FREE_GPU_MEM_FRACTION", [""])
@pytest.mark.parametrize("ENABLE_TRT_OVERLAP", ["False"],
                         ids=["disableTrtOverlap"])
@pytest.mark.parametrize("BATCHING_STRATEGY",
                         ["inflight_fused_batching", "V1"])
@pytest.mark.parametrize("DECOUPLED_MODE", ["False"],
                         ids=["disableDecoupleMode"])
@pytest.mark.parametrize("TRITON_MAX_BATCH_SIZE", ["128"])
@pytest.mark.parametrize("MAX_QUEUE_DELAY_MICROSECONDS", ["0"])
@pytest.mark.parametrize("MAX_BEAM_WIDTH", ["1", "4"])
@pytest.mark.parametrize("EXCLUDE_INPUT_IN_OUTPUT", ["False"])
def test_rcca_bug_4342666(MAX_NUM_SEQUENCE, MAX_TOKENS_IN_KV_CACHE,
                          BATCH_SCHEDULER_POLICY,
                          KV_CACHE_FREE_GPU_MEM_FRACTION, ENABLE_TRT_OVERLAP,
                          BATCHING_STRATEGY, DECOUPLED_MODE,
                          TRITON_MAX_BATCH_SIZE, MAX_QUEUE_DELAY_MICROSECONDS,
                          MAX_BEAM_WIDTH, EXCLUDE_INPUT_IN_OUTPUT,
                          inflight_batcher_llm_client_root,
                          llama_v2_tokenizer_model_root, llm_backend_venv):
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
        "tensorrt_llm/examples/llama/ib_llama_7b_chat_outputs")
    TOKENIZER_PATH = llama_v2_tokenizer_model_root
    TOKENIZER_TYPE = "llama"
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
        f"python3 {launch_server_py} --world_size=1 --model_repo={new_model_repo}",
        shell=True)
    check_server_ready()
    # Run Test
    TEXT = """
Input: Summarize the following conversation that took place between UberEats customer support......
Output:
Summarize the following conversation that took place between UberEats customer support and a customer:
Customer: Hi, I ordered food from UberEats an hour ago, but I still haven't received my order. Can you help me with this?
UberEats Support: Sorry to hear that. Can you please provide me with your order number so I can look into this for you?
Customer: Sure, it's #1234.
UberEats Support: Thank you. I've checked on your order, and it looks like the delivery partner is running a bit behind schedule. However, they should be arriving within the next 20 minutes. Would you like me to provide you with live updates on the status of your order?
Customer: Yes, that would be great. Can you also give me a discount on my order since it's taking so long?
UberEats Support: I understand your frustration. I can offer you a 10% discount on your order. Would you like me to apply that now?
Customer: Yes, that would be great. Thank you for your help.
UberEats Support: You're welcome. I've applied the discount to your order, and I'll make sure to provide you with live updates on the status of your delivery. Your order should arrive within the next 15 minutes. Is there anything else I can assist you with today?
Customer: No, that's all. Thank you for your help.
UberEats Support: You're welcome. Enjoy your meal!!
"""
    run_cmd = [
        f"{inflight_batcher_llm_client_root}/inflight_batcher_llm_client.py",
        f"--tokenizer-dir={llama_v2_tokenizer_model_root}",
        "--tokenizer-type=llama",
        "--request-output-len=500",
        f"--text={TEXT}",
    ]
    output_log = venv_check_output(llm_backend_venv, run_cmd)
    print_info(f"{output_log}")
    # Get output sentence from log
    m = re.search(r"Output beam 0:\s*(.*)\s*output_ids", output_log)
    output_result = ""
    if m is not None:
        output_result = m.group(1).strip()

    # Golden output sentence
    golden_result = """
In this conversation, the customer support representative was able to resolve the customer's issue by providing them with a discount on their order and keeping them updated on the status of their delivery. The representative was professional and courteous throughout the conversation, and the customer was satisfied with the resolution provided.
"""
    # Validate Accuracy
    threshold = 0.8
    validate_by_sequence_matcher(output_result, golden_result, threshold)
