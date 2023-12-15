import json
import os
import tempfile
import time
from difflib import SequenceMatcher

import pytest
from trt_test.misc import check_call, check_output, print_info

from .conftest import venv_check_call, venv_check_output


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
                           MAX_TOKENS_IN_KV_CACHE, MAX_ATTENTION_WINDOW_SIZE,
                           BATCH_SCHEDULER_POLICY, BATCHING_STRATEGY,
                           MAX_NUM_SEQUENCE, KV_CACHE_FREE_GPU_MEM_FRACTION,
                           EXCLUDE_INPUT_IN_OUTPUT, ENABLE_TRT_OVERLAP,
                           TRITON_MAX_BATCH_SIZE, MAX_QUEUE_DELAY_MICROSECONDS,
                           MAX_BEAM_WIDTH, PREPROCESSING_INSTANCE_COUNT,
                           POSTPROCESSING_INSTANCE_COUNT, ACCUMULATE_TOKEN,
                           BLS_INSTANCE_COUNT):
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
    tensorrt_llm_bls_config = os.path.join(llm_backend_repo_root,
                                           "triton_repo", "tensorrt_llm_bls",
                                           "config.pbtxt")
    check_call(
        f"python3 {fill_template_py} -i {llm_config} engine_dir:{ENGINE_PATH},decoupled_mode:{DECOUPLED_MODE}," \
        f"max_tokens_in_paged_kv_cache:{MAX_TOKENS_IN_KV_CACHE},max_attention_window_size:{MAX_ATTENTION_WINDOW_SIZE},batch_scheduler_policy:{BATCH_SCHEDULER_POLICY}," \
        f"batching_strategy:{BATCHING_STRATEGY},max_num_sequences:{MAX_NUM_SEQUENCE}," \
        f"kv_cache_free_gpu_mem_fraction:{KV_CACHE_FREE_GPU_MEM_FRACTION},enable_trt_overlap:{ENABLE_TRT_OVERLAP}," \
        f"exclude_input_in_output:{EXCLUDE_INPUT_IN_OUTPUT},triton_max_batch_size:{TRITON_MAX_BATCH_SIZE}," \
        f"max_queue_delay_microseconds:{MAX_QUEUE_DELAY_MICROSECONDS},max_beam_width:{MAX_BEAM_WIDTH}",
        shell=True)
    check_call(
        f"python3 {fill_template_py} -i {preprocessing_config} tokenizer_dir:{TOKENIZER_PATH},tokenizer_type:{TOKENIZER_TYPE}," \
        f"triton_max_batch_size:{TRITON_MAX_BATCH_SIZE},preprocessing_instance_count:{PREPROCESSING_INSTANCE_COUNT}",
        shell=True)
    check_call(
        f"python3 {fill_template_py} -i {postprocessing_config} tokenizer_dir:{TOKENIZER_PATH},tokenizer_type:{TOKENIZER_TYPE}," \
        f"triton_max_batch_size:{TRITON_MAX_BATCH_SIZE},postprocessing_instance_count:{POSTPROCESSING_INSTANCE_COUNT}",
        shell=True)
    check_call(
        f"python3 {fill_template_py} -i {ensemble_config} triton_max_batch_size:{TRITON_MAX_BATCH_SIZE}",
        shell=True)
    check_call(
        f"python3 {fill_template_py} -i {tensorrt_llm_bls_config} triton_max_batch_size:{TRITON_MAX_BATCH_SIZE}," \
        f"decoupled_mode:{DECOUPLED_MODE},accumulate_tokens:{ACCUMULATE_TOKEN},bls_instance_count:{BLS_INSTANCE_COUNT}",
        shell=True)


def validate_by_sequence_matcher(output_result, golden_results, threshold):
    rankings = {}
    for golden_result in golden_results:
        output_result = output_result.strip()
        golden_result = golden_result.strip()
        matcher = SequenceMatcher(None, output_result, golden_result)
        # Get the similarity ratio and populate rankings dict
        similarity_ratio = matcher.ratio()
        rankings[str(similarity_ratio)] = golden_result

    # Find out the highest_similarity_ratio
    highest_similarity_ratio, golden_result = max(rankings.items(),
                                                  key=lambda x: float(x[0]))
    print_info(f"output_result: {output_result}")
    print_info(
        f"rankings(similarity_ratio:golden_result):\n{json.dumps(rankings, indent=4)}"
    )

    if float(highest_similarity_ratio) < threshold:
        pytest.fail(
            f"highest_similarity_ratio {highest_similarity_ratio} is less than {threshold}"
        )


def run_cpp_backend_tests(feature_name, llm_backend_venv,
                          inflight_batcher_llm_client_root, tokenizer_dir,
                          tokenizer_type):
    # Chooses script
    script_name = ""
    if feature_name in ["test_basic", "test_log_probs"]:
        script_name = f"{inflight_batcher_llm_client_root}/inflight_batcher_llm_client.py"
    elif feature_name in ["test_stop_words", "test_embedding_bias"]:
        script_name = f"{inflight_batcher_llm_client_root}/end_to_end_grpc_client.py"

    # Run command
    if "inflight_batcher_llm_client.py" in script_name:
        run_cmd = [
            f"{script_name}",
            f"--tokenizer-dir={tokenizer_dir}",
            f"--tokenizer-type={tokenizer_type}",
        ]

        if feature_name == "test_basic":
            venv_check_call(llm_backend_venv, run_cmd)

        if feature_name == "test_log_probs":
            run_cmd += [
                "--request-output-len=10",
                "--return-log-probs",
                "--top-k=2",
            ]
            venv_check_call(llm_backend_venv, run_cmd)
    elif "end_to_end_grpc_client.py" in script_name:
        if feature_name == "test_stop_words":
            run_cmd = [
                f"{script_name}",
                f"-o=10",
                "-p=\"The only thing we have to fear is\"",
                "--stop-words=\" government\"",
            ]
            output = venv_check_output(llm_backend_venv, run_cmd)
            print_info(f"The test output is:\n{output}")
            with tempfile.NamedTemporaryFile(
                    dir=llm_backend_venv.get_working_directory(),
                    mode='w',
                    delete=False) as temp_file:
                temp_file.write(output)
                temp_file.close()
                check_call(
                    f"grep -v \"that the government will\" {temp_file.name}",
                    shell=True)
        if feature_name == "test_embedding_bias":
            run_cmd = [
                f"{script_name}",
                f"-o=10",
                "-p=\"The only thing we have to fear is\"",
                "--embedding-bias-words=\" government\"",
                "--embedding-bias-weights=-20",
            ]
            output = venv_check_output(llm_backend_venv, run_cmd)
            print_info(f"The test output is:\n{output}")
            with tempfile.NamedTemporaryFile(
                    dir=llm_backend_venv.get_working_directory(),
                    mode='w',
                    delete=False) as temp_file:
                temp_file.write(output)
                temp_file.close()
                check_call(
                    f"grep -v \"that the government will\" {temp_file.name}",
                    shell=True)


def run_cpp_streaming_backend_tests(feature_name, llm_backend_venv,
                                    inflight_batcher_llm_client_root,
                                    tokenizer_dir, tokenizer_type):
    # Chooses script
    script_name = ""
    if feature_name in ["test_basic"]:
        script_name = f"{inflight_batcher_llm_client_root}/inflight_batcher_llm_client.py"

    # Run command
    if "inflight_batcher_llm_client.py" in script_name:
        run_cmd = [
            f"{script_name}",
            "--streaming",
            f"--tokenizer-dir={tokenizer_dir}",
            f"--tokenizer-type={tokenizer_type}",
        ]

        if feature_name == "test_basic":
            venv_check_call(llm_backend_venv, run_cmd)
