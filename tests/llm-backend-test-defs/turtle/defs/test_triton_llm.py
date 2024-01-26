import os

import pytest
from trt_test.misc import call, check_call, print_info

from .build_engines import *
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


@pytest.mark.parametrize("E2E_MODEL_NAME", ["ensemble", "tensorrt_llm_bls"])
@pytest.mark.parametrize("ACCUMULATE_TOKEN", ["True", "False"])
@pytest.mark.parametrize("BLS_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("PREPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("POSTPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("MAX_TOKENS_IN_KV_CACHE", [""])
@pytest.mark.parametrize("MAX_ATTENTION_WINDOW_SIZE", [""])
@pytest.mark.parametrize("BATCH_SCHEDULER_POLICY",
                         ["max_utilization", "guaranteed_no_evict"])
@pytest.mark.parametrize("KV_CACHE_FREE_GPU_MEM_FRACTION", [""])
@pytest.mark.parametrize("ENABLE_TRT_OVERLAP", ["False"],
                         ids=["disableTrtOverlap"])
@pytest.mark.parametrize("BATCHING_STRATEGY",
                         ["inflight_fused_batching", "V1"])
@pytest.mark.parametrize("DECOUPLED_MODE", ["True", "False"],
                         ids=["enableDecoupleMode", "disableDecoupleMode"])
@pytest.mark.parametrize("TRITON_MAX_BATCH_SIZE", ["128"])
@pytest.mark.parametrize("MAX_QUEUE_DELAY_MICROSECONDS", ["0"])
@pytest.mark.parametrize("ENABLE_KV_CACHE_REUSE", ["False"])
@pytest.mark.parametrize("NORMALIZE_LOG_PROBS", ["True"])
@pytest.mark.parametrize("MAX_BEAM_WIDTH", ["1"])
@pytest.mark.parametrize("EXCLUDE_INPUT_IN_OUTPUT", ["False"])
@pytest.mark.parametrize("FEATURE_NAME", [
    "test_basic", "test_log_probs", "test_request_id", "test_stop_words",
    "test_embedding_bias"
])
def test_llama_v2_7b_ifb(
        E2E_MODEL_NAME, FEATURE_NAME, MAX_TOKENS_IN_KV_CACHE,
        MAX_ATTENTION_WINDOW_SIZE, BATCH_SCHEDULER_POLICY,
        KV_CACHE_FREE_GPU_MEM_FRACTION, ENABLE_TRT_OVERLAP, BATCHING_STRATEGY,
        DECOUPLED_MODE, TRITON_MAX_BATCH_SIZE, MAX_QUEUE_DELAY_MICROSECONDS,
        MAX_BEAM_WIDTH, ENABLE_KV_CACHE_REUSE, NORMALIZE_LOG_PROBS,
        PREPROCESSING_INSTANCE_COUNT, POSTPROCESSING_INSTANCE_COUNT,
        ACCUMULATE_TOKEN, BLS_INSTANCE_COUNT, EXCLUDE_INPUT_IN_OUTPUT,
        inflight_batcher_llm_client_root, tensorrt_llm_llama_example_root,
        llama_v2_tokenizer_model_root, llm_backend_venv):
    if BATCHING_STRATEGY == "V1" and BATCH_SCHEDULER_POLICY == "max_utilization":
        pytest.skip("Skipping. V1 doesn't support max_utilization.")

    if BATCHING_STRATEGY == "V1" and FEATURE_NAME == "test_embedding_bias":
        pytest.skip("Skipping. V1 doesn't support embedding_bias tensor yet.")

    if E2E_MODEL_NAME == "ensemble" and ACCUMULATE_TOKEN == "True":
        pytest.skip("Skipping.")

    llm_backend_repo_root = os.environ["LLM_BACKEND_ROOT"]
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
    modify_ib_config_pbtxt(
        new_model_repo, ENGINE_PATH, TOKENIZER_PATH, TOKENIZER_TYPE,
        llm_backend_repo_root, DECOUPLED_MODE, MAX_TOKENS_IN_KV_CACHE,
        MAX_ATTENTION_WINDOW_SIZE, BATCH_SCHEDULER_POLICY, BATCHING_STRATEGY,
        KV_CACHE_FREE_GPU_MEM_FRACTION, EXCLUDE_INPUT_IN_OUTPUT,
        ENABLE_TRT_OVERLAP, TRITON_MAX_BATCH_SIZE,
        MAX_QUEUE_DELAY_MICROSECONDS, MAX_BEAM_WIDTH, ENABLE_KV_CACHE_REUSE,
        NORMALIZE_LOG_PROBS, PREPROCESSING_INSTANCE_COUNT,
        POSTPROCESSING_INSTANCE_COUNT, ACCUMULATE_TOKEN, BLS_INSTANCE_COUNT)

    # Launch Triton Server
    launch_server_py = os.path.join(llm_backend_repo_root, "scripts",
                                    "launch_triton_server.py")
    check_call(
        f"python3 {launch_server_py} --world_size=1 --model_repo={new_model_repo}",
        shell=True)
    check_server_ready()
    # Run Test
    feature_name = f"{FEATURE_NAME}"
    tokenizer_dir = f"{llama_v2_tokenizer_model_root}"
    tokenizer_type = "llama"

    if DECOUPLED_MODE == "False":
        run_cpp_backend_tests(feature_name, llm_backend_venv,
                              inflight_batcher_llm_client_root, tokenizer_dir,
                              tokenizer_type)
    else:
        run_cpp_streaming_backend_tests(feature_name, llm_backend_venv,
                                        inflight_batcher_llm_client_root,
                                        tokenizer_dir, tokenizer_type)


@pytest.mark.parametrize("E2E_MODEL_NAME", ["ensemble"])
@pytest.mark.parametrize("ACCUMULATE_TOKEN", ["False"])
@pytest.mark.parametrize("BLS_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("PREPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("POSTPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("MAX_TOKENS_IN_KV_CACHE", [""])
@pytest.mark.parametrize("MAX_ATTENTION_WINDOW_SIZE", ["4096"])
@pytest.mark.parametrize("BATCH_SCHEDULER_POLICY",
                         ["max_utilization", "guaranteed_no_evict"])
@pytest.mark.parametrize("KV_CACHE_FREE_GPU_MEM_FRACTION", [""])
@pytest.mark.parametrize("ENABLE_TRT_OVERLAP", ["False"],
                         ids=["disableTrtOverlap"])
@pytest.mark.parametrize("BATCHING_STRATEGY",
                         ["inflight_fused_batching", "V1"])
@pytest.mark.parametrize("DECOUPLED_MODE", ["True", "False"],
                         ids=["enableDecoupleMode", "disableDecoupleMode"])
@pytest.mark.parametrize("TRITON_MAX_BATCH_SIZE", ["128"])
@pytest.mark.parametrize("MAX_QUEUE_DELAY_MICROSECONDS", ["0"])
@pytest.mark.parametrize("ENABLE_KV_CACHE_REUSE", ["False"])
@pytest.mark.parametrize("NORMALIZE_LOG_PROBS", ["True"])
@pytest.mark.parametrize("MAX_BEAM_WIDTH", ["1"])
@pytest.mark.parametrize("EXCLUDE_INPUT_IN_OUTPUT", ["False"])
def test_mistral_v1_7b_ifb(
        E2E_MODEL_NAME, MAX_TOKENS_IN_KV_CACHE, MAX_ATTENTION_WINDOW_SIZE,
        BATCH_SCHEDULER_POLICY, KV_CACHE_FREE_GPU_MEM_FRACTION,
        ENABLE_TRT_OVERLAP, BATCHING_STRATEGY, DECOUPLED_MODE,
        TRITON_MAX_BATCH_SIZE, MAX_QUEUE_DELAY_MICROSECONDS, MAX_BEAM_WIDTH,
        ENABLE_KV_CACHE_REUSE, NORMALIZE_LOG_PROBS,
        PREPROCESSING_INSTANCE_COUNT, POSTPROCESSING_INSTANCE_COUNT,
        ACCUMULATE_TOKEN, BLS_INSTANCE_COUNT, EXCLUDE_INPUT_IN_OUTPUT,
        inflight_batcher_llm_client_root, tensorrt_llm_llama_example_root,
        mistral_v1_tokenizer_model_root, llm_backend_venv):
    if BATCHING_STRATEGY == "V1" and BATCH_SCHEDULER_POLICY == "max_utilization":
        pytest.skip("Skipping. V1 doesn't support max_utilization.")

    if E2E_MODEL_NAME == "ensemble" and ACCUMULATE_TOKEN == "True":
        pytest.skip("Skipping.")

    llm_backend_repo_root = os.environ["LLM_BACKEND_ROOT"]
    # Build Engine
    ENGINE_PATH = prepare_mistral_v1_7b_engine(
        "ifb", tensorrt_llm_llama_example_root)

    # Prepare model repo
    new_model_repo = os.path.join(llm_backend_repo_root, "triton_repo")
    prepare_ib_model_repo(llm_backend_repo_root, new_model_repo)

    # Modify config.pbtxt
    TOKENIZER_PATH = mistral_v1_tokenizer_model_root
    TOKENIZER_TYPE = "llama"
    modify_ib_config_pbtxt(
        new_model_repo, ENGINE_PATH, TOKENIZER_PATH, TOKENIZER_TYPE,
        llm_backend_repo_root, DECOUPLED_MODE, MAX_TOKENS_IN_KV_CACHE,
        MAX_ATTENTION_WINDOW_SIZE, BATCH_SCHEDULER_POLICY, BATCHING_STRATEGY,
        KV_CACHE_FREE_GPU_MEM_FRACTION, EXCLUDE_INPUT_IN_OUTPUT,
        ENABLE_TRT_OVERLAP, TRITON_MAX_BATCH_SIZE,
        MAX_QUEUE_DELAY_MICROSECONDS, MAX_BEAM_WIDTH, ENABLE_KV_CACHE_REUSE,
        NORMALIZE_LOG_PROBS, PREPROCESSING_INSTANCE_COUNT,
        POSTPROCESSING_INSTANCE_COUNT, ACCUMULATE_TOKEN, BLS_INSTANCE_COUNT)

    # Launch Triton Server
    launch_server_py = os.path.join(llm_backend_repo_root, "scripts",
                                    "launch_triton_server.py")
    check_call(
        f"python3 {launch_server_py} --force --world_size 1 --model_repo={new_model_repo}",
        shell=True)
    check_server_ready()
    # Run Test
    run_cmd = [
        f"{inflight_batcher_llm_client_root}/inflight_batcher_llm_client.py",
        f"--tokenizer-dir={mistral_v1_tokenizer_model_root}",
        "--tokenizer-type=llama",
    ]
    if DECOUPLED_MODE == "True":
        run_cmd += [
            "--streaming",
        ]

    venv_check_call(llm_backend_venv, run_cmd)


@pytest.mark.parametrize("TEST_TYPE", ["e2e", "accuracy"])
@pytest.mark.parametrize("MAX_ATTENTION_WINDOW_SIZE", ["4096"])
def test_mistral_v1_7b_python_backend(TEST_TYPE, MAX_ATTENTION_WINDOW_SIZE,
                                      llm_backend_gpt_example_root,
                                      mistral_v1_tokenizer_model_root,
                                      tensorrt_llm_llama_example_root,
                                      llm_backend_venv):
    llm_backend_repo_root = os.environ["LLM_BACKEND_ROOT"]
    # Build Engine
    ENGINE_PATH = prepare_mistral_v1_7b_engine(
        "python_backend", tensorrt_llm_llama_example_root)
    # Prepare model repo
    origin_model_repo = os.path.join(llm_backend_repo_root, "all_models",
                                     "gpt")
    new_model_repo = os.path.join(llm_backend_repo_root, "triton_repo")
    check_call(f"rm -rf {new_model_repo}", shell=True)
    check_call(f"cp -R {origin_model_repo} {new_model_repo}", shell=True)

    # Modify config.pbtxt
    TOKENIZER_PATH = mistral_v1_tokenizer_model_root
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
        f"python3 {fill_template_py} -i {llm_config} engine_dir:{ENGINE_PATH},max_attention_window_size:{MAX_ATTENTION_WINDOW_SIZE}",
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


@pytest.mark.skip_less_device(8)
@pytest.mark.parametrize("E2E_MODEL_NAME", ["ensemble"])
@pytest.mark.parametrize("ACCUMULATE_TOKEN", ["False"])
@pytest.mark.parametrize("BLS_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("PREPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("POSTPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("MAX_TOKENS_IN_KV_CACHE", [""])
@pytest.mark.parametrize("MAX_ATTENTION_WINDOW_SIZE", [""])
@pytest.mark.parametrize("BATCH_SCHEDULER_POLICY",
                         ["max_utilization", "guaranteed_no_evict"])
@pytest.mark.parametrize("KV_CACHE_FREE_GPU_MEM_FRACTION", [""])
@pytest.mark.parametrize("ENABLE_TRT_OVERLAP", ["False"],
                         ids=["disableTrtOverlap"])
@pytest.mark.parametrize("BATCHING_STRATEGY",
                         ["inflight_fused_batching", "V1"])
@pytest.mark.parametrize("DECOUPLED_MODE", ["True", "False"],
                         ids=["enableDecoupleMode", "disableDecoupleMode"])
@pytest.mark.parametrize("TRITON_MAX_BATCH_SIZE", ["128"])
@pytest.mark.parametrize("MAX_QUEUE_DELAY_MICROSECONDS", ["0"])
@pytest.mark.parametrize("ENABLE_KV_CACHE_REUSE", ["False"])
@pytest.mark.parametrize("NORMALIZE_LOG_PROBS", ["True"])
@pytest.mark.parametrize("MAX_BEAM_WIDTH", ["1"])
@pytest.mark.parametrize("EXCLUDE_INPUT_IN_OUTPUT", ["False"])
def test_llama_v2_70b_ifb(
        E2E_MODEL_NAME, MAX_TOKENS_IN_KV_CACHE, MAX_ATTENTION_WINDOW_SIZE,
        BATCH_SCHEDULER_POLICY, KV_CACHE_FREE_GPU_MEM_FRACTION,
        ENABLE_TRT_OVERLAP, BATCHING_STRATEGY, DECOUPLED_MODE,
        TRITON_MAX_BATCH_SIZE, MAX_QUEUE_DELAY_MICROSECONDS, MAX_BEAM_WIDTH,
        ENABLE_KV_CACHE_REUSE, NORMALIZE_LOG_PROBS,
        PREPROCESSING_INSTANCE_COUNT, POSTPROCESSING_INSTANCE_COUNT,
        ACCUMULATE_TOKEN, BLS_INSTANCE_COUNT, EXCLUDE_INPUT_IN_OUTPUT,
        inflight_batcher_llm_client_root, tensorrt_llm_llama_example_root,
        llama_v2_tokenizer_model_root, llm_backend_venv):
    if BATCHING_STRATEGY == "V1" and BATCH_SCHEDULER_POLICY == "max_utilization":
        pytest.skip("Skipping. V1 doesn't support max_utilization.")

    if E2E_MODEL_NAME == "ensemble" and ACCUMULATE_TOKEN == "True":
        pytest.skip("Skipping.")

    llm_backend_repo_root = os.environ["LLM_BACKEND_ROOT"]
    # Build Engine
    ENGINE_PATH = prepare_llama_v2_70b_engine("ifb",
                                              tensorrt_llm_llama_example_root,
                                              llama_v2_tokenizer_model_root)
    # Prepare model repo
    new_model_repo = os.path.join(llm_backend_repo_root, "triton_repo")
    prepare_ib_model_repo(llm_backend_repo_root, new_model_repo)

    # Modify config.pbtxt
    TOKENIZER_PATH = llama_v2_tokenizer_model_root
    TOKENIZER_TYPE = "llama"
    modify_ib_config_pbtxt(
        new_model_repo, ENGINE_PATH, TOKENIZER_PATH, TOKENIZER_TYPE,
        llm_backend_repo_root, DECOUPLED_MODE, MAX_TOKENS_IN_KV_CACHE,
        MAX_ATTENTION_WINDOW_SIZE, BATCH_SCHEDULER_POLICY, BATCHING_STRATEGY,
        KV_CACHE_FREE_GPU_MEM_FRACTION, EXCLUDE_INPUT_IN_OUTPUT,
        ENABLE_TRT_OVERLAP, TRITON_MAX_BATCH_SIZE,
        MAX_QUEUE_DELAY_MICROSECONDS, MAX_BEAM_WIDTH, ENABLE_KV_CACHE_REUSE,
        NORMALIZE_LOG_PROBS, PREPROCESSING_INSTANCE_COUNT,
        POSTPROCESSING_INSTANCE_COUNT, ACCUMULATE_TOKEN, BLS_INSTANCE_COUNT)

    # Launch Triton Server
    launch_server_py = os.path.join(llm_backend_repo_root, "scripts",
                                    "launch_triton_server.py")
    check_call(
        f"python3 {launch_server_py} --world_size=8 --model_repo={new_model_repo}",
        shell=True)
    check_server_ready()
    # Run Test
    run_cmd = [
        f"{inflight_batcher_llm_client_root}/inflight_batcher_llm_client.py",
        f"--tokenizer-dir={llama_v2_tokenizer_model_root}",
        "--tokenizer-type=llama",
    ]
    if DECOUPLED_MODE == "True":
        run_cmd += [
            "--streaming",
        ]

    venv_check_call(llm_backend_venv, run_cmd)


@pytest.mark.parametrize("TEST_TYPE", ["e2e", "accuracy"])
def test_gpt_350m_python_backend(TEST_TYPE, llm_backend_gpt_example_root,
                                 tensorrt_llm_gpt_example_root,
                                 gpt_tokenizer_model_root, llm_backend_venv):
    llm_backend_repo_root = os.environ["LLM_BACKEND_ROOT"]
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


@pytest.mark.parametrize("E2E_MODEL_NAME", ["ensemble", "tensorrt_llm_bls"])
@pytest.mark.parametrize("ACCUMULATE_TOKEN", ["True", "False"])
@pytest.mark.parametrize("BLS_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("PREPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("POSTPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("MAX_TOKENS_IN_KV_CACHE", [""])
@pytest.mark.parametrize("MAX_ATTENTION_WINDOW_SIZE", [""])
@pytest.mark.parametrize("BATCH_SCHEDULER_POLICY",
                         ["max_utilization", "guaranteed_no_evict"])
@pytest.mark.parametrize("KV_CACHE_FREE_GPU_MEM_FRACTION", [""])
@pytest.mark.parametrize("ENABLE_TRT_OVERLAP", ["False"],
                         ids=["disableTrtOverlap"])
@pytest.mark.parametrize("BATCHING_STRATEGY",
                         ["inflight_fused_batching", "V1"])
@pytest.mark.parametrize("DECOUPLED_MODE", ["True", "False"],
                         ids=["enableDecoupleMode", "disableDecoupleMode"])
@pytest.mark.parametrize("TRITON_MAX_BATCH_SIZE", ["128"])
@pytest.mark.parametrize("MAX_QUEUE_DELAY_MICROSECONDS", ["0"])
@pytest.mark.parametrize("ENABLE_KV_CACHE_REUSE", ["False"])
@pytest.mark.parametrize("NORMALIZE_LOG_PROBS", ["True"])
@pytest.mark.parametrize("MAX_BEAM_WIDTH", ["1"])
@pytest.mark.parametrize("EXCLUDE_INPUT_IN_OUTPUT", ["False"])
@pytest.mark.parametrize("FEATURE_NAME", [
    "test_basic", "test_log_probs", "test_request_id", "test_stop_words",
    "test_embedding_bias"
])
def test_gpt_350m_ifb(
        E2E_MODEL_NAME, FEATURE_NAME, MAX_TOKENS_IN_KV_CACHE,
        MAX_ATTENTION_WINDOW_SIZE, BATCH_SCHEDULER_POLICY,
        KV_CACHE_FREE_GPU_MEM_FRACTION, ENABLE_TRT_OVERLAP, BATCHING_STRATEGY,
        DECOUPLED_MODE, TRITON_MAX_BATCH_SIZE, MAX_QUEUE_DELAY_MICROSECONDS,
        MAX_BEAM_WIDTH, ENABLE_KV_CACHE_REUSE, NORMALIZE_LOG_PROBS,
        PREPROCESSING_INSTANCE_COUNT, POSTPROCESSING_INSTANCE_COUNT,
        ACCUMULATE_TOKEN, BLS_INSTANCE_COUNT, EXCLUDE_INPUT_IN_OUTPUT,
        inflight_batcher_llm_client_root, tensorrt_llm_gpt_example_root,
        gpt_tokenizer_model_root, llm_backend_venv):
    if BATCHING_STRATEGY == "V1" and BATCH_SCHEDULER_POLICY == "max_utilization":
        pytest.skip("Skipping. V1 doesn't support max_utilization.")

    if BATCHING_STRATEGY == "V1" and FEATURE_NAME == "test_embedding_bias":
        pytest.skip("Skipping. V1 doesn't support embedding_bias tensor yet.")

    if E2E_MODEL_NAME == "ensemble" and ACCUMULATE_TOKEN == "True":
        pytest.skip("Skipping.")

    llm_backend_repo_root = os.environ["LLM_BACKEND_ROOT"]
    # Build engine
    ENGINE_PATH = prepare_gpt_350m_engine(
        "ifb",
        tensorrt_llm_gpt_example_root,
        gpt_tokenizer_model_root,
    )
    # Prepare model repo
    new_model_repo = os.path.join(llm_backend_repo_root, "triton_repo")
    prepare_ib_model_repo(llm_backend_repo_root, new_model_repo)

    # Modify config.pbtxt
    TOKENIZER_PATH = gpt_tokenizer_model_root
    TOKENIZER_TYPE = "auto"
    modify_ib_config_pbtxt(
        new_model_repo, ENGINE_PATH, TOKENIZER_PATH, TOKENIZER_TYPE,
        llm_backend_repo_root, DECOUPLED_MODE, MAX_TOKENS_IN_KV_CACHE,
        MAX_ATTENTION_WINDOW_SIZE, BATCH_SCHEDULER_POLICY, BATCHING_STRATEGY,
        KV_CACHE_FREE_GPU_MEM_FRACTION, EXCLUDE_INPUT_IN_OUTPUT,
        ENABLE_TRT_OVERLAP, TRITON_MAX_BATCH_SIZE,
        MAX_QUEUE_DELAY_MICROSECONDS, MAX_BEAM_WIDTH, ENABLE_KV_CACHE_REUSE,
        NORMALIZE_LOG_PROBS, PREPROCESSING_INSTANCE_COUNT,
        POSTPROCESSING_INSTANCE_COUNT, ACCUMULATE_TOKEN, BLS_INSTANCE_COUNT)

    # Launch Triton Server
    launch_server_py = os.path.join(llm_backend_repo_root, "scripts",
                                    "launch_triton_server.py")
    check_call(
        f"python3 {launch_server_py} --world_size=1 --model_repo={new_model_repo}",
        shell=True)
    check_server_ready()
    # Run Test
    feature_name = f"{FEATURE_NAME}"
    tokenizer_dir = f"{gpt_tokenizer_model_root}"
    tokenizer_type = "auto"

    if DECOUPLED_MODE == "False":
        run_cpp_backend_tests(feature_name, llm_backend_venv,
                              inflight_batcher_llm_client_root, tokenizer_dir,
                              tokenizer_type)
    else:
        run_cpp_streaming_backend_tests(feature_name, llm_backend_venv,
                                        inflight_batcher_llm_client_root,
                                        tokenizer_dir, tokenizer_type)


@pytest.mark.parametrize("E2E_MODEL_NAME", ["ensemble"])
@pytest.mark.parametrize("ACCUMULATE_TOKEN", ["False"])
@pytest.mark.parametrize("BLS_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("PREPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("POSTPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("MAX_TOKENS_IN_KV_CACHE", [""])
@pytest.mark.parametrize("MAX_ATTENTION_WINDOW_SIZE", [""])
@pytest.mark.parametrize("BATCH_SCHEDULER_POLICY",
                         ["max_utilization", "guaranteed_no_evict"])
@pytest.mark.parametrize("KV_CACHE_FREE_GPU_MEM_FRACTION", [""])
@pytest.mark.parametrize("ENABLE_TRT_OVERLAP", ["False"],
                         ids=["disableTrtOverlap"])
@pytest.mark.parametrize("BATCHING_STRATEGY",
                         ["inflight_fused_batching", "V1"])
@pytest.mark.parametrize("DECOUPLED_MODE", ["False"],
                         ids=["disableDecoupleMode"])
@pytest.mark.parametrize("TRITON_MAX_BATCH_SIZE", ["128"])
@pytest.mark.parametrize("MAX_QUEUE_DELAY_MICROSECONDS", ["0"])
@pytest.mark.parametrize("ENABLE_KV_CACHE_REUSE", ["True"])
@pytest.mark.parametrize("NORMALIZE_LOG_PROBS", ["True"])
@pytest.mark.parametrize("MAX_BEAM_WIDTH", ["1"])
@pytest.mark.parametrize("EXCLUDE_INPUT_IN_OUTPUT", ["False"])
def test_gpt_350m_speculative_decoding(
        E2E_MODEL_NAME, MAX_TOKENS_IN_KV_CACHE, MAX_ATTENTION_WINDOW_SIZE,
        BATCH_SCHEDULER_POLICY, KV_CACHE_FREE_GPU_MEM_FRACTION,
        ENABLE_TRT_OVERLAP, BATCHING_STRATEGY, DECOUPLED_MODE,
        TRITON_MAX_BATCH_SIZE, MAX_QUEUE_DELAY_MICROSECONDS, MAX_BEAM_WIDTH,
        ENABLE_KV_CACHE_REUSE, NORMALIZE_LOG_PROBS,
        PREPROCESSING_INSTANCE_COUNT, POSTPROCESSING_INSTANCE_COUNT,
        ACCUMULATE_TOKEN, BLS_INSTANCE_COUNT, EXCLUDE_INPUT_IN_OUTPUT,
        tensorrt_llm_gpt_example_root, gpt_tokenizer_model_root,
        gpt2_medium_tokenizer_model_root,
        llm_backend_inflight_batcher_llm_root, llm_backend_dataset_root,
        llm_backend_venv):
    if BATCHING_STRATEGY == "V1":
        pytest.skip("Skipping. Speculative decoding is not supported in V1.")

    if E2E_MODEL_NAME == "ensemble" and ACCUMULATE_TOKEN == "True":
        pytest.skip("Skipping.")

    llm_backend_repo_root = os.environ["LLM_BACKEND_ROOT"]
    # Build engine
    ENGINE_PATH = prepare_gpt_350m_engine(
        "medium_ifb",
        tensorrt_llm_gpt_example_root,
        gpt2_medium_tokenizer_model_root,
    )
    DRAFT_ENGINE_PATH = prepare_gpt_350m_engine(
        "ifb",
        tensorrt_llm_gpt_example_root,
        gpt_tokenizer_model_root,
    )
    # Prepare two model repos
    ## first repo
    new_model_repo = os.path.join(llm_backend_repo_root, "triton_repo")
    prepare_ib_model_repo(llm_backend_repo_root, new_model_repo)
    ## second repo
    new_model_repo_draft = os.path.join(llm_backend_repo_root,
                                        "triton_repo_draft")
    prepare_ib_model_repo(llm_backend_repo_root, new_model_repo_draft)

    # Modify two suits of config.pbtxt
    ## first suit
    TOKENIZER_PATH = gpt2_medium_tokenizer_model_root
    TOKENIZER_TYPE = "auto"
    modify_ib_config_pbtxt(
        new_model_repo, ENGINE_PATH, TOKENIZER_PATH, TOKENIZER_TYPE,
        llm_backend_repo_root, DECOUPLED_MODE, MAX_TOKENS_IN_KV_CACHE,
        MAX_ATTENTION_WINDOW_SIZE, BATCH_SCHEDULER_POLICY, BATCHING_STRATEGY,
        KV_CACHE_FREE_GPU_MEM_FRACTION, EXCLUDE_INPUT_IN_OUTPUT,
        ENABLE_TRT_OVERLAP, TRITON_MAX_BATCH_SIZE,
        MAX_QUEUE_DELAY_MICROSECONDS, MAX_BEAM_WIDTH, ENABLE_KV_CACHE_REUSE,
        NORMALIZE_LOG_PROBS, PREPROCESSING_INSTANCE_COUNT,
        POSTPROCESSING_INSTANCE_COUNT, ACCUMULATE_TOKEN, BLS_INSTANCE_COUNT)
    ## second suit
    TOKENIZER_PATH = gpt_tokenizer_model_root
    TOKENIZER_TYPE = "auto"
    ENABLE_KV_CACHE_REUSE = "False"
    modify_ib_config_pbtxt(
        new_model_repo_draft, DRAFT_ENGINE_PATH, TOKENIZER_PATH,
        TOKENIZER_TYPE, llm_backend_repo_root, DECOUPLED_MODE,
        MAX_TOKENS_IN_KV_CACHE, MAX_ATTENTION_WINDOW_SIZE,
        BATCH_SCHEDULER_POLICY, BATCHING_STRATEGY,
        KV_CACHE_FREE_GPU_MEM_FRACTION, EXCLUDE_INPUT_IN_OUTPUT,
        ENABLE_TRT_OVERLAP, TRITON_MAX_BATCH_SIZE,
        MAX_QUEUE_DELAY_MICROSECONDS, MAX_BEAM_WIDTH, ENABLE_KV_CACHE_REUSE,
        NORMALIZE_LOG_PROBS, PREPROCESSING_INSTANCE_COUNT,
        POSTPROCESSING_INSTANCE_COUNT, ACCUMULATE_TOKEN, BLS_INSTANCE_COUNT)
    # Launch two Triton Servers
    ## first server
    launch_server_py = os.path.join(llm_backend_repo_root, "scripts",
                                    "launch_triton_server.py")
    check_call(
        f"python3 {launch_server_py} --world_size=1 --model_repo={new_model_repo}",
        shell=True)
    check_server_ready(http_port="8000")
    ## second server
    launch_server_py = os.path.join(llm_backend_repo_root, "scripts",
                                    "launch_triton_server.py")
    check_call(
        f"python3 {launch_server_py} --world_size=1 --model_repo={new_model_repo} " \
        f"--grpc_port=8004 --http_port=8003 --metrics_port=8005",
        shell=True)
    check_server_ready(http_port="8003")
    # Run Test
    run_cmd = [
        f"{llm_backend_inflight_batcher_llm_root}/speculative_decoding_test.py",
        "--max-input-len=200",
        f"--dataset={llm_backend_dataset_root}/mini_cnn_eval.json",
        "--url-draft=0.0.0.0:8004",
        "--url-target=0.0.0.0:8001",
    ]

    venv_check_call(llm_backend_venv, run_cmd)


@pytest.mark.skip_less_device(8)
@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.parametrize("E2E_MODEL_NAME", ["ensemble"])
@pytest.mark.parametrize("ACCUMULATE_TOKEN", ["False"])
@pytest.mark.parametrize("BLS_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("PREPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("POSTPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("MAX_TOKENS_IN_KV_CACHE", [""])
@pytest.mark.parametrize("MAX_ATTENTION_WINDOW_SIZE", [""])
@pytest.mark.parametrize("BATCH_SCHEDULER_POLICY",
                         ["max_utilization", "guaranteed_no_evict"])
@pytest.mark.parametrize("KV_CACHE_FREE_GPU_MEM_FRACTION", [""])
@pytest.mark.parametrize("ENABLE_TRT_OVERLAP", ["False"],
                         ids=["disableTrtOverlap"])
@pytest.mark.parametrize("BATCHING_STRATEGY",
                         ["inflight_fused_batching", "V1"])
@pytest.mark.parametrize("DECOUPLED_MODE", ["True", "False"],
                         ids=["enableDecoupleMode", "disableDecoupleMode"])
@pytest.mark.parametrize("TRITON_MAX_BATCH_SIZE", ["128"])
@pytest.mark.parametrize("MAX_QUEUE_DELAY_MICROSECONDS", ["0"])
@pytest.mark.parametrize("ENABLE_KV_CACHE_REUSE", ["False"])
@pytest.mark.parametrize("NORMALIZE_LOG_PROBS", ["True"])
@pytest.mark.parametrize("MAX_BEAM_WIDTH", ["1"])
@pytest.mark.parametrize("EXCLUDE_INPUT_IN_OUTPUT", ["False"])
def test_gpt_175b_ifb(
        E2E_MODEL_NAME, MAX_TOKENS_IN_KV_CACHE, MAX_ATTENTION_WINDOW_SIZE,
        BATCH_SCHEDULER_POLICY, KV_CACHE_FREE_GPU_MEM_FRACTION,
        ENABLE_TRT_OVERLAP, BATCHING_STRATEGY, DECOUPLED_MODE,
        TRITON_MAX_BATCH_SIZE, MAX_QUEUE_DELAY_MICROSECONDS, MAX_BEAM_WIDTH,
        ENABLE_KV_CACHE_REUSE, NORMALIZE_LOG_PROBS,
        PREPROCESSING_INSTANCE_COUNT, POSTPROCESSING_INSTANCE_COUNT,
        ACCUMULATE_TOKEN, BLS_INSTANCE_COUNT, EXCLUDE_INPUT_IN_OUTPUT,
        inflight_batcher_llm_client_root, tensorrt_llm_gpt_example_root,
        gpt_tokenizer_model_root, llm_backend_venv):
    if BATCHING_STRATEGY == "V1" and BATCH_SCHEDULER_POLICY == "max_utilization":
        pytest.skip("Skipping. V1 doesn't support max_utilization.")

    if E2E_MODEL_NAME == "ensemble" and ACCUMULATE_TOKEN == "True":
        pytest.skip("Skipping.")

    llm_backend_repo_root = os.environ["LLM_BACKEND_ROOT"]
    # Build Engine
    ENGINE_PATH = prepare_gpt_175b_engine("ifb", tensorrt_llm_gpt_example_root)
    # Prepare model repo
    new_model_repo = os.path.join(llm_backend_repo_root, "triton_repo")
    prepare_ib_model_repo(llm_backend_repo_root, new_model_repo)

    # Modify config.pbtxt
    TOKENIZER_PATH = gpt_tokenizer_model_root
    TOKENIZER_TYPE = "auto"
    modify_ib_config_pbtxt(
        new_model_repo, ENGINE_PATH, TOKENIZER_PATH, TOKENIZER_TYPE,
        llm_backend_repo_root, DECOUPLED_MODE, MAX_TOKENS_IN_KV_CACHE,
        MAX_ATTENTION_WINDOW_SIZE, BATCH_SCHEDULER_POLICY, BATCHING_STRATEGY,
        KV_CACHE_FREE_GPU_MEM_FRACTION, EXCLUDE_INPUT_IN_OUTPUT,
        ENABLE_TRT_OVERLAP, TRITON_MAX_BATCH_SIZE,
        MAX_QUEUE_DELAY_MICROSECONDS, MAX_BEAM_WIDTH, ENABLE_KV_CACHE_REUSE,
        NORMALIZE_LOG_PROBS, PREPROCESSING_INSTANCE_COUNT,
        POSTPROCESSING_INSTANCE_COUNT, ACCUMULATE_TOKEN, BLS_INSTANCE_COUNT)

    # Launch Triton Server
    launch_server_py = os.path.join(llm_backend_repo_root, "scripts",
                                    "launch_triton_server.py")
    check_call(
        f"python3 {launch_server_py} --world_size=8 --model_repo={new_model_repo}",
        shell=True)
    check_server_ready()
    # Run Test
    run_cmd = [
        f"{inflight_batcher_llm_client_root}/inflight_batcher_llm_client.py",
        f"--tokenizer-dir={gpt_tokenizer_model_root}",
        "--tokenizer-type=auto",
    ]
    if DECOUPLED_MODE == "True":
        run_cmd += [
            "--streaming",
        ]

    venv_check_call(llm_backend_venv, run_cmd)


@pytest.mark.parametrize("E2E_MODEL_NAME", ["ensemble"])
@pytest.mark.parametrize("ACCUMULATE_TOKEN", ["False"])
@pytest.mark.parametrize("BLS_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("PREPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("POSTPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("MAX_TOKENS_IN_KV_CACHE", [""])
@pytest.mark.parametrize("MAX_ATTENTION_WINDOW_SIZE", [""])
@pytest.mark.parametrize("BATCH_SCHEDULER_POLICY", ["guaranteed_no_evict"])
@pytest.mark.parametrize("KV_CACHE_FREE_GPU_MEM_FRACTION", [""])
@pytest.mark.parametrize("ENABLE_TRT_OVERLAP", ["False"],
                         ids=["disableTrtOverlap"])
@pytest.mark.parametrize("BATCHING_STRATEGY", ["inflight_fused_batching"])
@pytest.mark.parametrize("DECOUPLED_MODE", ["False"],
                         ids=["disableDecoupleMode"])
@pytest.mark.parametrize("TRITON_MAX_BATCH_SIZE", ["128"])
@pytest.mark.parametrize("MAX_QUEUE_DELAY_MICROSECONDS", ["0"])
@pytest.mark.parametrize("ENABLE_KV_CACHE_REUSE", ["False"])
@pytest.mark.parametrize("NORMALIZE_LOG_PROBS", ["True"])
@pytest.mark.parametrize("MAX_BEAM_WIDTH", ["1"])
@pytest.mark.parametrize("EXCLUDE_INPUT_IN_OUTPUT", ["False"])
@pytest.mark.parametrize("VIRTUAL_TOKENS", ["True", "False"],
                         ids=["withVirtualTokens", "withoutVirtualTokens"])
def test_gpt_next_ptuning_ifb(
        E2E_MODEL_NAME, MAX_TOKENS_IN_KV_CACHE, MAX_ATTENTION_WINDOW_SIZE,
        BATCH_SCHEDULER_POLICY, KV_CACHE_FREE_GPU_MEM_FRACTION,
        ENABLE_TRT_OVERLAP, BATCHING_STRATEGY, DECOUPLED_MODE,
        TRITON_MAX_BATCH_SIZE, MAX_QUEUE_DELAY_MICROSECONDS, MAX_BEAM_WIDTH,
        ENABLE_KV_CACHE_REUSE, NORMALIZE_LOG_PROBS,
        PREPROCESSING_INSTANCE_COUNT, POSTPROCESSING_INSTANCE_COUNT,
        ACCUMULATE_TOKEN, BLS_INSTANCE_COUNT, EXCLUDE_INPUT_IN_OUTPUT,
        VIRTUAL_TOKENS, inflight_batcher_llm_client_root,
        gpt_tokenizer_model_root, tensorrt_llm_example_root,
        tensorrt_llm_gpt_example_root, gpt_next_ptuning_model_root,
        llm_backend_venv):
    if BATCHING_STRATEGY == "V1" and BATCH_SCHEDULER_POLICY == "max_utilization":
        pytest.skip("Skipping. V1 doesn't support max_utilization.")

    if E2E_MODEL_NAME == "ensemble" and ACCUMULATE_TOKEN == "True":
        pytest.skip("Skipping.")

    llm_backend_repo_root = os.environ["LLM_BACKEND_ROOT"]
    # Build engine
    ENGINE_PATH, output_model_dir = prepare_gpt_next_ptuning_engine(
        "ifb", tensorrt_llm_gpt_example_root, gpt_next_ptuning_model_root)
    # Prepare model repo
    new_model_repo = os.path.join(llm_backend_repo_root, "triton_repo")
    prepare_ib_model_repo(llm_backend_repo_root, new_model_repo)

    # Modify config.pbtxt
    TOKENIZER_PATH = gpt_tokenizer_model_root
    TOKENIZER_TYPE = "auto"
    modify_ib_config_pbtxt(
        new_model_repo, ENGINE_PATH, TOKENIZER_PATH, TOKENIZER_TYPE,
        llm_backend_repo_root, DECOUPLED_MODE, MAX_TOKENS_IN_KV_CACHE,
        MAX_ATTENTION_WINDOW_SIZE, BATCH_SCHEDULER_POLICY, BATCHING_STRATEGY,
        KV_CACHE_FREE_GPU_MEM_FRACTION, EXCLUDE_INPUT_IN_OUTPUT,
        ENABLE_TRT_OVERLAP, TRITON_MAX_BATCH_SIZE,
        MAX_QUEUE_DELAY_MICROSECONDS, MAX_BEAM_WIDTH, ENABLE_KV_CACHE_REUSE,
        NORMALIZE_LOG_PROBS, PREPROCESSING_INSTANCE_COUNT,
        POSTPROCESSING_INSTANCE_COUNT, ACCUMULATE_TOKEN, BLS_INSTANCE_COUNT)

    # Generate reference output
    run_py_path = os.path.join(tensorrt_llm_example_root, "run.py")
    vocab_file = os.path.join(output_model_dir, "1-gpu", "tokenizer.model")
    # 1. Input with virtual tokens:
    if VIRTUAL_TOKENS == "True":
        prompt_table = os.path.join(tensorrt_llm_gpt_example_root,
                                    "email_composition.npy")
        input_tokens = os.path.join(tensorrt_llm_gpt_example_root, "input.csv")
        run_cmd = [
            f"{run_py_path}",
            "--max_output_len=8",
            f"--vocab_file={vocab_file}",
            f"--prompt_table_path={prompt_table}",
            f"--input_file={input_tokens}",
            f"--engine_dir={ENGINE_PATH}",
            f"--output_csv=output_w_prompt.csv",
            "--no_add_special_tokens",
        ]
        venv_check_call(llm_backend_venv, run_cmd)
    # 2. Input w/o virtual tokens:
    elif VIRTUAL_TOKENS == "False":
        input_wo_prompt_csv = os.path.join(
            llm_backend_venv.get_working_directory(), "input_wo_prompt.csv")
        check_call(
            f"echo \"25229,291,7379,251522,39854,5754,251514,315,32906,14297,398,261\" > {input_wo_prompt_csv}",
            shell=True)
        run_cmd = [
            f"{run_py_path}",
            "--max_output_len=8",
            f"--vocab_file={vocab_file}",
            f"--input_file={input_wo_prompt_csv}",
            f"--engine_dir={ENGINE_PATH}",
            f"--output_csv=output_wo_prompt.csv",
            "--no_add_special_tokens",
        ]
        venv_check_call(llm_backend_venv, run_cmd)

    # Launch Triton Server
    launch_server_py = os.path.join(llm_backend_repo_root, "scripts",
                                    "launch_triton_server.py")
    check_call(
        f"python3 {launch_server_py} --world_size=1 --model_repo={new_model_repo}",
        shell=True)
    check_server_ready()

    # Run Test
    if VIRTUAL_TOKENS == "True":
        run_cmd = [
            f"{inflight_batcher_llm_client_root}/inflight_batcher_llm_client.py",
            f"--prompt-embedding-table={prompt_table}", "--prompt-task-id=0",
            f"--input-tokens-csv={input_tokens}",
            "--output-tokens-csv=output_w_prompt.csv",
            "--request-output-len=8", "--check-output"
        ]
        venv_check_call(llm_backend_venv, run_cmd)
    elif VIRTUAL_TOKENS == "False":
        run_cmd = [
            f"{inflight_batcher_llm_client_root}/inflight_batcher_llm_client.py",
            f"--input-tokens-csv={input_wo_prompt_csv}",
            "--output-tokens-csv=output_wo_prompt.csv",
            "--request-output-len=8", "--check-output"
        ]
        venv_check_call(llm_backend_venv, run_cmd)


@pytest.mark.parametrize("E2E_MODEL_NAME", ["ensemble"])
@pytest.mark.parametrize("ACCUMULATE_TOKEN", ["False"])
@pytest.mark.parametrize("BLS_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("PREPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("POSTPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("MAX_TOKENS_IN_KV_CACHE", [""])
@pytest.mark.parametrize("MAX_ATTENTION_WINDOW_SIZE", [""])
@pytest.mark.parametrize("BATCH_SCHEDULER_POLICY", ["guaranteed_no_evict"])
@pytest.mark.parametrize("KV_CACHE_FREE_GPU_MEM_FRACTION", [""])
@pytest.mark.parametrize("ENABLE_TRT_OVERLAP", ["False"],
                         ids=["disableTrtOverlap"])
@pytest.mark.parametrize("BATCHING_STRATEGY", ["inflight_fused_batching"])
@pytest.mark.parametrize("DECOUPLED_MODE", ["False"],
                         ids=["disableDecoupleMode"])
@pytest.mark.parametrize("TRITON_MAX_BATCH_SIZE", ["128"])
@pytest.mark.parametrize("MAX_QUEUE_DELAY_MICROSECONDS", ["0"])
@pytest.mark.parametrize("ENABLE_KV_CACHE_REUSE", ["False"])
@pytest.mark.parametrize("NORMALIZE_LOG_PROBS", ["True"])
@pytest.mark.parametrize("MAX_BEAM_WIDTH", ["1"])
@pytest.mark.parametrize("EXCLUDE_INPUT_IN_OUTPUT", ["False"])
def test_gpt_2b_lora_ifb(
        E2E_MODEL_NAME, MAX_TOKENS_IN_KV_CACHE, MAX_ATTENTION_WINDOW_SIZE,
        BATCH_SCHEDULER_POLICY, KV_CACHE_FREE_GPU_MEM_FRACTION,
        ENABLE_TRT_OVERLAP, BATCHING_STRATEGY, DECOUPLED_MODE,
        TRITON_MAX_BATCH_SIZE, MAX_QUEUE_DELAY_MICROSECONDS, MAX_BEAM_WIDTH,
        ENABLE_KV_CACHE_REUSE, NORMALIZE_LOG_PROBS,
        PREPROCESSING_INSTANCE_COUNT, POSTPROCESSING_INSTANCE_COUNT,
        ACCUMULATE_TOKEN, BLS_INSTANCE_COUNT, EXCLUDE_INPUT_IN_OUTPUT,
        inflight_batcher_llm_client_root, tensorrt_llm_example_root,
        tensorrt_llm_gpt_example_root, gpt_2b_lora_model_root, models_root,
        llm_backend_venv):
    if BATCHING_STRATEGY == "V1":
        pytest.skip("Skipping. LoRA is not supported in V1.")

    if E2E_MODEL_NAME == "ensemble" and ACCUMULATE_TOKEN == "True":
        pytest.skip("Skipping.")

    llm_backend_repo_root = os.environ["LLM_BACKEND_ROOT"]
    # Build engine
    ENGINE_PATH = prepare_gpt_2b_lora_engine("ifb",
                                             tensorrt_llm_gpt_example_root,
                                             gpt_2b_lora_model_root,
                                             models_root)
    # Prepare model repo
    new_model_repo = os.path.join(llm_backend_repo_root, "triton_repo")
    prepare_ib_model_repo(llm_backend_repo_root, new_model_repo)

    # Modify config.pbtxt
    TOKENIZER_PATH = os.path.join(models_root, "gpt-next",
                                  "gpt-next-tokenizer-hf-v2")
    TOKENIZER_TYPE = "auto"
    modify_ib_config_pbtxt(
        new_model_repo, ENGINE_PATH, TOKENIZER_PATH, TOKENIZER_TYPE,
        llm_backend_repo_root, DECOUPLED_MODE, MAX_TOKENS_IN_KV_CACHE,
        MAX_ATTENTION_WINDOW_SIZE, BATCH_SCHEDULER_POLICY, BATCHING_STRATEGY,
        KV_CACHE_FREE_GPU_MEM_FRACTION, EXCLUDE_INPUT_IN_OUTPUT,
        ENABLE_TRT_OVERLAP, TRITON_MAX_BATCH_SIZE,
        MAX_QUEUE_DELAY_MICROSECONDS, MAX_BEAM_WIDTH, ENABLE_KV_CACHE_REUSE,
        NORMALIZE_LOG_PROBS, PREPROCESSING_INSTANCE_COUNT,
        POSTPROCESSING_INSTANCE_COUNT, ACCUMULATE_TOKEN, BLS_INSTANCE_COUNT)

    # Generate reference output
    run_py_path = os.path.join(tensorrt_llm_example_root, "run.py")
    # Input with virtual tokens:
    input_tokens = os.path.join(tensorrt_llm_gpt_example_root, "input.csv")
    output_tokens = os.path.join(tensorrt_llm_gpt_example_root, "output.csv")
    lora_path = os.path.join(tensorrt_llm_gpt_example_root,
                             "gpt-2b-lora-train-900")
    lora_tllm_path = os.path.join(tensorrt_llm_gpt_example_root,
                                  "gpt-2b-lora-train-900-tllm")
    run_cmd = [
        f"{run_py_path}",
        "--max_output_len=8",
        f"--lora_dir={lora_tllm_path}",
        "--lora_ckpt_source=nemo",
        "--lora_task_uids=lora",
        f"--input_file={input_tokens}",
        f"--output_csv={output_tokens}",
        f"--engine_dir={ENGINE_PATH}",
        "--use_py_session",
    ]
    venv_check_call(llm_backend_venv, run_cmd)

    # Launch Triton Server
    launch_server_py = os.path.join(llm_backend_repo_root, "scripts",
                                    "launch_triton_server.py")
    check_call(
        f"python3 {launch_server_py} --world_size=1 --model_repo={new_model_repo}",
        shell=True)
    check_server_ready()

    # Run Test
    run_cmd = [
        f"{inflight_batcher_llm_client_root}/inflight_batcher_llm_client.py",
        f"--input-tokens-csv={input_tokens}",
        f"--output-tokens-csv={output_tokens}",
        "--request-output-len=8",
        "--check-output",
        f"--lora-path={lora_path}",
    ]
    venv_check_call(llm_backend_venv, run_cmd)
