# -*- coding: utf-8 -*-

import os
import shutil
import tempfile

import pytest
# Conftest imports require defs root. This is not the case inside test defines however.
from trt_test.misc import check_call, check_output, print_info
from trt_test.session_data_writer import SessionDataWriter

pytest_plugins = ["pytester", "trt_test.pytest_plugin"]


def llm_models_root() -> str:
    '''return LLM_MODELS_ROOT path if it is set in env, assert when it's set but not a valid path
    '''
    LLM_MODELS_ROOT = os.environ.get("LLM_MODELS_ROOT", None)
    if LLM_MODELS_ROOT is not None:
        assert os.path.isabs(
            LLM_MODELS_ROOT), "LLM_MODELS_ROOT must be absolute path"
        assert os.path.exists(
            LLM_MODELS_ROOT), "LLM_MODELS_ROOT must exists when its specified"
    return LLM_MODELS_ROOT


def venv_check_call(venv, cmd):

    def _war_check_call(*args, **kwargs):
        kwargs["cwd"] = venv.get_working_directory()
        return check_call(*args, **kwargs)

    venv.run_cmd(cmd, caller=_war_check_call, print_script=False)


def venv_check_output(venv, cmd):

    def _war_check_output(*args, **kwargs):
        kwargs["cwd"] = venv.get_working_directory()
        output = check_output(*args, **kwargs)
        return output

    return venv.run_cmd(cmd, caller=_war_check_output, print_script=False)


@pytest.fixture(scope="session")
def trt_py3_venv_factory(trt_py_base_venv_factory):
    """
    Session-scoped fixture which provides a factory function to produce a VirtualenvRunner capable of
    running Python3 code.  Used by other session-scoped fixtures which need to modify the default VirtualenvRunner prolog.
    """

    # TODO: remove update env after TURTLE support multi devices
    # Temporarily update CUDA_VISIBLE_DEVICES visible device
    device_count = get_device_count()
    visible_devices = ",".join([str(i) for i in range(device_count)])

    print_info(f"Setting CUDA_VISIBLE_DEVICES to {visible_devices}.")

    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices

    def factory():
        return trt_py_base_venv_factory("python3")

    return factory


@pytest.fixture(scope="session")
def llm_backend_root(trt_py3_venv_factory):
    assert "LLM_BACKEND_ROOT" in os.environ, "You must set LLM_BACKEND_ROOT env to the root dir of the TRT LLM Backend repo to start turtle tests"
    llm_backend_repo_root = os.environ["LLM_BACKEND_ROOT"]
    assert os.path.isabs(
        llm_backend_repo_root
    ), f"LLM_BACKEND_ROOT should be abs path, got: {llm_backend_repo_root}"
    assert os.path.exists(
        llm_backend_repo_root
    ), f"LLM_BACKEND_ROOT should exists, got:{llm_backend_repo_root}"
    return llm_backend_repo_root


@pytest.fixture(scope="session")
def trt_performance_cache_name():
    return "performance.cache"


@pytest.fixture(scope="session")
def trt_performance_cache_fpath(trt_config, trt_performance_cache_name):
    fpath = os.path.join(trt_config["workspace"], trt_performance_cache_name)
    return fpath


# Get the executing turtle case name
@pytest.fixture(autouse=True)
def turtle_case_name(request):
    return request.node.nodeid


@pytest.fixture(scope="session")
def output_dir(request):
    return request.config._trt_config["output_dir"]


@pytest.fixture(scope="session")
def llm_session_data_writer(trt_config, trt_gpu_clock_lock,
                            versions_from_infer_device, output_dir):
    """
    Fixture for the SessionDataWriter, used to write session data to output directory.
    """

    # Attempt to see if we can run infer_device to get the necessary tags for perf_runner
    perf_tag_data = trt_config["perf_trt_tag"]

    if versions_from_infer_device:
        for k, v in versions_from_infer_device.items():
            if k not in perf_tag_data or perf_tag_data[k] is None:
                perf_tag_data[k] = v

    session_data_writer = SessionDataWriter(
        perf_trt_tag=perf_tag_data,
        log_output_directory=output_dir,
        output_formats=trt_config["perf_log_formats"],
        gpu_clock_lock=trt_gpu_clock_lock,
    )

    yield session_data_writer

    session_data_writer.teardown()


@pytest.fixture(scope="session")
def llm_backend_venv(trt_py3_venv_factory):
    """
    The fixture venv used for LLM tests.
    """
    venv = trt_py3_venv_factory()
    return venv


@pytest.fixture(scope="session")
def llm_backend_gpt_example_root(llm_backend_root):
    backend_gpt_example_root = os.path.join(llm_backend_root, "tools", "gpt")
    return backend_gpt_example_root


@pytest.fixture(scope="session")
def llm_backend_multimodal_example_root(llm_backend_root):
    backend_multimodal_example_root = os.path.join(llm_backend_root, "tools",
                                                   "multimodal")
    return backend_multimodal_example_root


@pytest.fixture(scope="session")
def llm_backend_inflight_batcher_llm_root(llm_backend_root):
    backend_gpt_example_root = os.path.join(llm_backend_root, "tools",
                                            "inflight_batcher_llm")
    return backend_gpt_example_root


@pytest.fixture(scope="session")
def llm_backend_dataset_root(llm_backend_root):
    backend_gpt_example_root = os.path.join(llm_backend_root, "tools",
                                            "dataset")
    return backend_gpt_example_root


@pytest.fixture(scope="session")
def tensorrt_llm_example_root(llm_backend_root):
    llm_gpt_example_root = os.path.join(llm_backend_root, "tensorrt_llm",
                                        "examples")
    return llm_gpt_example_root


@pytest.fixture(scope="session")
def tensorrt_llm_gpt_example_root(llm_backend_root):
    llm_gpt_example_root = os.path.join(llm_backend_root, "tensorrt_llm",
                                        "examples", "gpt")
    return llm_gpt_example_root


@pytest.fixture(scope="session")
def tensorrt_llm_gptj_example_root(llm_backend_root):
    llm_gpt_example_root = os.path.join(llm_backend_root, "tensorrt_llm",
                                        "examples", "gptj")
    return llm_gpt_example_root


@pytest.fixture(scope="session")
def tensorrt_llm_multimodal_example_root(llm_backend_root):
    llm_multimodal_example_root = os.path.join(llm_backend_root,
                                               "tensorrt_llm", "examples",
                                               "multimodal")
    return llm_multimodal_example_root


@pytest.fixture(scope="session")
def tensorrt_llm_opt_example_root(llm_backend_root):
    llm_opt_example_root = os.path.join(llm_backend_root, "tensorrt_llm",
                                        "examples", "opt")
    return llm_opt_example_root


@pytest.fixture(scope="session")
def tensorrt_llm_medusa_example_root(llm_backend_root):
    llm_medusa_example_root = os.path.join(llm_backend_root, "tensorrt_llm",
                                           "examples", "medusa")
    return llm_medusa_example_root


@pytest.fixture(scope="session")
def tensorrt_llm_enc_dec_example_root(llm_backend_root):
    llm_enc_dec_example_root = os.path.join(llm_backend_root, "tensorrt_llm",
                                            "examples", "enc_dec")
    return llm_enc_dec_example_root


@pytest.fixture(scope="session")
def tensorrt_llm_llama_example_root(llm_backend_root):
    llm_llama_example_root = os.path.join(llm_backend_root, "tensorrt_llm",
                                          "examples", "llama")
    return llm_llama_example_root


@pytest.fixture(scope="session")
def inflight_batcher_llm_client_root(llm_backend_root):
    inflight_batcher_llm_client_root = os.path.join(llm_backend_root,
                                                    "inflight_batcher_llm",
                                                    "client")

    assert os.path.exists(
        inflight_batcher_llm_client_root
    ), f"{inflight_batcher_llm_client_root} does not exists."
    return inflight_batcher_llm_client_root


@pytest.fixture(autouse=True)
def skip_by_device_count(request):
    if request.node.get_closest_marker('skip_less_device'):
        device_count = get_device_count()
        expected_count = request.node.get_closest_marker(
            'skip_less_device').args[0]
        if expected_count > int(device_count):
            pytest.skip(
                f'Device count {device_count} is less than {expected_count}')


def get_device_count():
    output = check_output("nvidia-smi -L", shell=True, cwd="/tmp")
    device_count = len(output.strip().split('\n'))

    return device_count


@pytest.fixture(autouse=True)
def skip_by_device_memory(request):
    "fixture for skip less device memory"
    if request.node.get_closest_marker('skip_less_device_memory'):
        device_memory = get_device_memory()
        expected_memory = request.node.get_closest_marker(
            'skip_less_device_memory').args[0]
        if expected_memory > int(device_memory):
            pytest.skip(
                f'Device memory {device_memory} is less than {expected_memory}'
            )


def get_device_memory():
    "get gpu memory"
    memory = 0
    with tempfile.TemporaryDirectory() as temp_dirname:
        cmd = " ".join([
            "nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader"
        ])
        output = check_output(cmd, shell=True, cwd=temp_dirname)
        memory = int(output.strip().split()[0])

    return memory


@pytest.fixture(scope="session")
def models_root():
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"

    return models_root


@pytest.fixture(scope="session")
def llama_v2_tokenizer_model_root():
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"
    llama_v2_tokenizer_model_root = os.path.join(models_root,
                                                 "llama-models-v2")

    assert os.path.exists(
        llama_v2_tokenizer_model_root
    ), f"{llama_v2_tokenizer_model_root} does not exist under NFS LLM_MODELS_ROOT dir"
    return llama_v2_tokenizer_model_root


@pytest.fixture(scope="session")
def mistral_v1_tokenizer_model_root():
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"
    mistral_v1_tokenizer_model_root = os.path.join(models_root,
                                                   "mistral-7b-v0.1")

    assert os.path.exists(
        mistral_v1_tokenizer_model_root
    ), f"{mistral_v1_tokenizer_model_root} does not exist under NFS LLM_MODELS_ROOT dir"
    return mistral_v1_tokenizer_model_root


@pytest.fixture(scope="session")
def gpt_tokenizer_model_root(llm_backend_venv):
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"
    gpt_tokenizer_model_root = os.path.join(models_root, "gpt2")

    assert os.path.exists(
        gpt_tokenizer_model_root
    ), f"{gpt_tokenizer_model_root} does not exist under NFS LLM_MODELS_ROOT dir"
    return gpt_tokenizer_model_root


@pytest.fixture(scope="session")
def gptj_tokenizer_model_root(llm_backend_venv):
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"
    gptj_tokenizer_model_root = os.path.join(models_root, "gpt-j-6b")

    assert os.path.exists(
        gptj_tokenizer_model_root
    ), f"{gptj_tokenizer_model_root} does not exist under NFS LLM_MODELS_ROOT dir"
    return gptj_tokenizer_model_root


@pytest.fixture(scope="session")
def gpt2_medium_tokenizer_model_root():
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"
    gpt_tokenizer_model_root = os.path.join(models_root, "gpt2-medium")

    assert os.path.exists(
        gpt_tokenizer_model_root
    ), f"{gpt_tokenizer_model_root} does not exist under NFS LLM_MODELS_ROOT dir"
    return gpt_tokenizer_model_root


@pytest.fixture(scope="session")
def gpt_next_ptuning_model_root():
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"
    gpt_next_ptuning_model_root = os.path.join(models_root,
                                               "email_composition")

    assert os.path.exists(
        gpt_next_ptuning_model_root
    ), f"{gpt_next_ptuning_model_root} does not exist under NFS LLM_MODELS_ROOT dir"
    return gpt_next_ptuning_model_root


@pytest.fixture(scope="session")
def gpt_2b_lora_model_root():
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"
    gpt_2b_lora_model_root = os.path.join(models_root, "lora", "gpt-next-2b")

    assert os.path.exists(
        gpt_2b_lora_model_root
    ), f"{gpt_2b_lora_model_root} does not exist under NFS LLM_MODELS_ROOT dir"
    return gpt_2b_lora_model_root


@pytest.fixture(scope="session")
def blip2_opt_model_root():
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"
    blip2_opt_model_root = os.path.join(models_root, "blip2-opt-2.7b")

    assert os.path.exists(
        blip2_opt_model_root
    ), f"{blip2_opt_model_root} does not exist under NFS LLM_MODELS_ROOT dir"
    return blip2_opt_model_root


@pytest.fixture(scope="session")
def llava_model_root():
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"
    llava_model_root = os.path.join(models_root, "llava-1.5-7b-hf")

    assert os.path.exists(
        llava_model_root
    ), f"{llava_model_root} does not exist under NFS LLM_MODELS_ROOT dir"
    return llava_model_root


@pytest.fixture(scope="session")
def llama_v3_8b_model_root():
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"
    llama_model_root = os.path.join(models_root, "llama-models-v3",
                                    "llama-v3-8b-instruct-hf")

    assert os.path.exists(
        llama_model_root
    ), f"{llama_model_root} does not exist under NFS LLM_MODELS_ROOT dir"
    return llama_model_root


@pytest.fixture(scope="session")
def llama_v3_70b_model_root():
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"
    llama_model_root = os.path.join(models_root, "llama-models-v3",
                                    "Llama-3-70B-Instruct-Gradient-1048k")

    assert os.path.exists(
        llama_model_root
    ), f"{llama_model_root} does not exist under NFS LLM_MODELS_ROOT dir"
    return llama_model_root


@pytest.fixture(scope="session")
def vicuna_7b_model_root():
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"
    vicuna_7b_model_root = os.path.join(models_root, "vicuna-7b-v1.3")

    assert os.path.exists(
        vicuna_7b_model_root
    ), f"{vicuna_7b_model_root} does not exist under NFS LLM_MODELS_ROOT dir"
    return vicuna_7b_model_root


@pytest.fixture(scope="session")
def medusa_vicuna_7b_model_root():
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"
    medusa_vicuna_7b_model_root = os.path.join(models_root,
                                               "medusa-vicuna-7b-v1.3")

    assert os.path.exists(
        medusa_vicuna_7b_model_root
    ), f"{medusa_vicuna_7b_model_root} does not exist under NFS LLM_MODELS_ROOT dir"
    return medusa_vicuna_7b_model_root


@pytest.fixture(scope="session")
def t5_small_model_root():
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"
    t5_small_model_root = os.path.join(models_root, "t5-small")

    assert os.path.exists(
        t5_small_model_root
    ), f"{t5_small_model_root} does not exist under NFS LLM_MODELS_ROOT dir"
    return t5_small_model_root


# Returns an array of total memory for each available device
@pytest.fixture(scope="session")
def total_gpu_memory_mib():
    output = check_output("nvidia-smi --query-gpu memory.total --format=csv",
                          shell=True,
                          cwd="/tmp")
    lines = [l.strip() for l in output.strip().split("\n")]
    lines = lines[1:]  # skip header
    lines = [l[:-4] for l in lines]  # remove MiB suffix
    lines = [int(l) for l in lines]
    return lines


# Pytset cache mechanism can be used to store and retrieve data across test runs.
@pytest.fixture(scope="session", autouse=True)
def setup_cache_data(request, tensorrt_llm_example_root):
    # This variable will be used in hook function: pytest_runtest_teardown since
    # fixtures cannot be directly used in hooks.
    request.config.cache.set('example_root', tensorrt_llm_example_root)


def cleanup_engine_outputs(output_dir_root):
    for dirpath, dirnames, _ in os.walk(output_dir_root, topdown=False):
        for dirname in dirnames:
            if "engine_dir" in dirname or "model_dir" in dirname or "ckpt_dir" in dirname:
                folder_path = os.path.join(dirpath, dirname)
                try:
                    shutil.rmtree(folder_path)
                    print_info(f"Deleted folder: {folder_path}")
                except Exception as e:
                    print_info(f"Error deleting {folder_path}: {e}")


# Teardown hook to clean up engine outputs after each group of test cases are finished
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item, nextitem):
    current_test_basename = item.name.split(
        "[")[0] if '[' in item.name else item.name

    if nextitem:
        next_test_basename = nextitem.name.split(
            "[")[0] if '[' in nextitem.name else nextitem.name
    else:
        next_test_basename = None

    if next_test_basename != current_test_basename:
        print_info("Cleaning up engine outputs:")
        engine_outputs_root = item.config.cache.get('example_root', None)
        cleanup_engine_outputs(engine_outputs_root)

    yield
