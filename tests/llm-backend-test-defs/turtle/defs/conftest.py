# -*- coding: utf-8 -*-

import os

import pytest
# Conftest imports require defs root. This is not the case inside test defines however.
from trt_test.misc import call, check_call, check_output, print_info

pytest_plugins = ["pytester", "trt_test.pytest_plugin"]


def venv_check_call(venv, cmd):

    def _war_check_call(*args, **kwargs):
        kwargs["cwd"] = venv.get_working_directory()
        return check_call(*args, **kwargs)

    venv.run_cmd(cmd, caller=_war_check_call)


def venv_check_output(venv, cmd):

    def _war_check_output(*args, **kwargs):
        kwargs["cwd"] = venv.get_working_directory()
        output = check_output(*args, **kwargs)
        return output

    return venv.run_cmd(cmd, caller=_war_check_output)


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


# Get the executing turtle case name
@pytest.fixture(autouse=True)
def turtle_case_name(request):
    return request.node.nodeid


@pytest.fixture(scope="session")
def output_dir(request):
    return request.config._trt_config["output_dir"]


@pytest.fixture(scope="session")
def llm_backend_venv(trt_py3_venv_factory):
    """
    The fixture venv used for LLM tests.
    """
    venv = trt_py3_venv_factory()
    return venv


@pytest.fixture(scope="session")
def llm_backend_gpt_example_root(llm_backend_root, llm_backend_venv):
    backend_gpt_example_root = os.path.join(llm_backend_root, "tools", "gpt")
    workspace = llm_backend_venv.get_working_directory()

    check_call([
        "wget", "-q",
        "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json",
        "--directory-prefix", workspace
    ])
    check_call([
        "wget", "-q",
        "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt",
        "--directory-prefix", workspace
    ])

    return backend_gpt_example_root


@pytest.fixture(scope="function")
def engine_dir(llm_backend_venv):
    "Get engine dir"
    engine_path = os.path.join(llm_backend_venv.get_working_directory(),
                               "engines")

    yield engine_path

    call(f"rm -rf {engine_path}", shell=True)


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
