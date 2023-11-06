import os
import time

import pytest
from trt_test.misc import check_call, check_output, print_info

from .conftest import venv_check_call, venv_check_output


@pytest.fixture(autouse=True)
def stop_triton_server():
    # Stop Triton Server after each test
    yield
    check_call(f"pkill tritonserver", shell=True)
    time.sleep(5)


def check_server_ready():
    timeout = 1200
    timer = 0
    while True:
        status = check_output(
            r"curl -s -w %{http_code} 0.0.0.0:8000/v2/health/ready || true",
            shell=True).strip()
        if status == "200":
            break
        elif timer <= timeout:
            time.sleep(5)
            timer += 5
        elif timer > timeout:
            raise TimeoutError("Error: Launch Triton server timed out.")

    print_info(
        f"Trion server launched successfully! Cost {timer} seconds to launch server."
    )


MAX_NUM_SEQUENCES = (""
                     "32")
MAX_TOKENS_IN_KV_CACHES = ("")
BATCH_SCHEDULER_POLICIES = ("max_utilization"
                            "guaranteed_no_evict")
KV_CACHE_FREE_GPU_MEM_FRACTIONS = (""
                                   "0.2")


@pytest.mark.parametrize("MAX_NUM_SEQUENCE", [""])
@pytest.mark.parametrize("MAX_TOKENS_IN_KV_CACHE", [""])
@pytest.mark.parametrize("BATCH_SCHEDULER_POLICY",
                         ["max_utilization", "guaranteed_no_evict"])
@pytest.mark.parametrize("KV_CACHE_FREE_GPU_MEM_FRACTION", [""])
@pytest.mark.parametrize("ENABLE_TRT_OVERLAP", ["False"],
                         ids=["disableTrtOverlap"])
@pytest.mark.parametrize("BATCHING_STRATEGY",
                         ["inflight_fused_batching", "V1"])
@pytest.mark.parametrize("DECOUPLED_MODE", ["True", "False"],
                         ids=["enableDecoupleMode", "disableDecoupleMode"])
def test_llama_v2_7b_ib(MAX_NUM_SEQUENCE, MAX_TOKENS_IN_KV_CACHE,
                        BATCH_SCHEDULER_POLICY, KV_CACHE_FREE_GPU_MEM_FRACTION,
                        ENABLE_TRT_OVERLAP, BATCHING_STRATEGY, DECOUPLED_MODE,
                        inflight_batcher_llm_client_root,
                        llama_v2_tokenizer_model_root, llm_backend_venv):
    llm_backend_repo_root = os.environ["LLM_BACKEND_ROOT"]
    # Prepare model repo
    origin_model_repo = os.path.join(llm_backend_repo_root, "all_models",
                                     "inflight_batcher_llm")
    new_model_repo = os.path.join(llm_backend_repo_root, "triton_repo")
    check_call(f"rm -rf {new_model_repo}", shell=True)
    check_call(f"cp -R {origin_model_repo} {new_model_repo}", shell=True)

    # Modify config.pbtxt
    ENGINE_PATH = os.path.join(
        llm_backend_repo_root,
        "tensorrt_llm/examples/llama/ib_llama_7b_outputs")
    TOKENIZER_PATH = llama_v2_tokenizer_model_root
    TOKENIZER_TYPE = "llama"
    fill_template_py = os.path.join(llm_backend_repo_root, "tools",
                                    "fill_template.py")
    llm_config = os.path.join(llm_backend_repo_root, "triton_repo",
                              "tensorrt_llm", "config.pbtxt")
    preprocessing_config = os.path.join(llm_backend_repo_root, "triton_repo",
                                        "preprocessing", "config.pbtxt")
    postprocessing_config = os.path.join(llm_backend_repo_root, "triton_repo",
                                         "postprocessing", "config.pbtxt")
    check_call(
        f"python3 {fill_template_py} -i {llm_config} engine_dir:{ENGINE_PATH},decoupled_mode:{DECOUPLED_MODE}," \
        f"max_tokens_in_paged_kv_cache:{MAX_TOKENS_IN_KV_CACHE},batch_scheduler_policy:{BATCH_SCHEDULER_POLICY}," \
        f"batching_strategy:{BATCHING_STRATEGY},max_num_sequences:{MAX_NUM_SEQUENCE}," \
        f"kv_cache_free_gpu_mem_fraction:{KV_CACHE_FREE_GPU_MEM_FRACTION},enable_trt_overlap:{ENABLE_TRT_OVERLAP}",
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
        f"python3 {launch_server_py} --world_size 1 --model_repo={new_model_repo}",
        shell=True)
    check_server_ready()
    # Run Test
    run_cmd = [
        f"{inflight_batcher_llm_client_root}/inflight_batcher_llm_client.py",
        f"--tokenizer_dir={llama_v2_tokenizer_model_root}",
        "--tokenizer_type=llama",
    ]
    if DECOUPLED_MODE == "True":
        run_cmd += [
            "--streaming",
        ]

    venv_check_call(llm_backend_venv, run_cmd)


@pytest.mark.parametrize("TEST_TYPE", ["e2e", "accuracy"])
def test_gpt_350m_normal(TEST_TYPE, llm_backend_gpt_example_root,
                         gpt_tokenizer_model_root, llm_backend_venv):
    llm_backend_repo_root = os.environ["LLM_BACKEND_ROOT"]
    # Prepare model repo
    origin_model_repo = os.path.join(llm_backend_repo_root, "all_models",
                                     "gpt")
    new_model_repo = os.path.join(llm_backend_repo_root, "triton_repo")
    check_call(f"rm -rf {new_model_repo}", shell=True)
    check_call(f"cp -R {origin_model_repo} {new_model_repo}", shell=True)
    # Modify config.pbtxt
    ENGINE_PATH = os.path.join(
        llm_backend_repo_root,
        "tensorrt_llm/examples/gpt/trt_engine/gpt2/fp16/1-gpu")
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
        f"python3 {launch_server_py} --world_size 1 --model_repo={new_model_repo}",
        shell=True)
    check_server_ready()
    # Run Test
    if TEST_TYPE == "e2e":
        run_cmd = [
            f"{llm_backend_gpt_example_root}/end_to_end_test.py",
            f"--tokenizer_dir={TOKENIZER_PATH}",
            f"--tokenizer_type={TOKENIZER_TYPE}",
        ]
        venv_check_call(llm_backend_venv, run_cmd)
    elif TEST_TYPE == "accuracy":
        run_cmd = [
            f"{llm_backend_gpt_example_root}/client.py",
            "--text=Born in north-east France, Soyer trained as a",
            "--output_len=10",
            f"--tokenizer_dir={TOKENIZER_PATH}",
            f"--tokenizer_type={TOKENIZER_TYPE}",
        ]

        output = venv_check_output(llm_backend_venv,
                                   run_cmd).strip().split("\n")[-1]

        print_info(output)
        # Validate Accuracy -ToDo
