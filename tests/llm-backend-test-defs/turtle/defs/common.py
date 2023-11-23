import os
import time
from difflib import SequenceMatcher

import pytest
from trt_test.misc import check_call, check_output, print_info


def check_server_ready():
    timeout = 600
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


def prepare_ib_model_repo(llm_backend_repo_root, new_model_repo):
    origin_model_repo = os.path.join(llm_backend_repo_root, "all_models",
                                     "inflight_batcher_llm")
    check_call(f"rm -rf {new_model_repo}", shell=True)
    check_call(f"cp -R {origin_model_repo} {new_model_repo}", shell=True)


def modify_ib_config_pbtxt(ENGINE_PATH, TOKENIZER_PATH, TOKENIZER_TYPE,
                           llm_backend_repo_root, DECOUPLED_MODE,
                           MAX_TOKENS_IN_KV_CACHE, MAX_KV_CACHE_LEN,
                           BATCH_SCHEDULER_POLICY, BATCHING_STRATEGY,
                           MAX_NUM_SEQUENCE, KV_CACHE_FREE_GPU_MEM_FRACTION,
                           EXCLUDE_INPUT_IN_OUTPUT, ENABLE_TRT_OVERLAP,
                           TRITON_MAX_BATCH_SIZE, MAX_QUEUE_DELAY_MICROSECONDS,
                           MAX_BEAM_WIDTH):
    fill_template_py = os.path.join(llm_backend_repo_root, "tools",
                                    "fill_template.py")
    llm_config = os.path.join(llm_backend_repo_root, "triton_repo",
                              "tensorrt_llm", "config.pbtxt")
    preprocessing_config = os.path.join(llm_backend_repo_root, "triton_repo",
                                        "preprocessing", "config.pbtxt")
    postprocessing_config = os.path.join(llm_backend_repo_root, "triton_repo",
                                         "postprocessing", "config.pbtxt")
    ensemble_config = os.path.join(llm_backend_repo_root, "triton_repo",
                                   "ensemble", "config.pbtxt")
    check_call(
        f"python3 {fill_template_py} -i {llm_config} engine_dir:{ENGINE_PATH},decoupled_mode:{DECOUPLED_MODE}," \
        f"max_tokens_in_paged_kv_cache:{MAX_TOKENS_IN_KV_CACHE},max_kv_cache_length:{MAX_KV_CACHE_LEN},batch_scheduler_policy:{BATCH_SCHEDULER_POLICY}," \
        f"batching_strategy:{BATCHING_STRATEGY},max_num_sequences:{MAX_NUM_SEQUENCE}," \
        f"kv_cache_free_gpu_mem_fraction:{KV_CACHE_FREE_GPU_MEM_FRACTION},enable_trt_overlap:{ENABLE_TRT_OVERLAP}," \
        f"exclude_input_in_output:{EXCLUDE_INPUT_IN_OUTPUT},triton_max_batch_size:{TRITON_MAX_BATCH_SIZE}," \
        f"max_queue_delay_microseconds:{MAX_QUEUE_DELAY_MICROSECONDS},max_batch_width:{MAX_BEAM_WIDTH}",
        shell=True)
    check_call(
        f"python3 {fill_template_py} -i {preprocessing_config} tokenizer_dir:{TOKENIZER_PATH},tokenizer_type:{TOKENIZER_TYPE},triton_max_batch_size:{TRITON_MAX_BATCH_SIZE}",
        shell=True)
    check_call(
        f"python3 {fill_template_py} -i {postprocessing_config} tokenizer_dir:{TOKENIZER_PATH},tokenizer_type:{TOKENIZER_TYPE},triton_max_batch_size:{TRITON_MAX_BATCH_SIZE}",
        shell=True)
    check_call(
        f"python3 {fill_template_py} -i {ensemble_config} triton_max_batch_size:{TRITON_MAX_BATCH_SIZE}",
        shell=True)


def validate_by_sequence_matcher(output_result, golden_result, threshold):
    output_result = output_result.strip()
    golden_result = golden_result.strip()
    matcher = SequenceMatcher(None, output_result, golden_result)
    # Get the similarity ratio
    similarity_ratio = matcher.ratio()
    print_info(f"output_result: {output_result}")
    print_info(f"golden_result: {golden_result}")
    print_info(f"similarity_ratio: {similarity_ratio}")

    if similarity_ratio < threshold:
        pytest.fail(
            f"similarity_ratio {similarity_ratio} is less than {threshold}")
