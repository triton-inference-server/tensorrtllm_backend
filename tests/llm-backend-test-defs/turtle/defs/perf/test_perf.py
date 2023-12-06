"""
TensorRT LLM Backend perf tests
"""
import os
import re
from typing import Dict, List

import pytest

from .utils import (AbstractPerfScriptTestClass, PerfMetricType,
                    PerfScriptTestCmd)


class Metric:

    def __init__(self, tp: PerfMetricType, regexp):
        self.tp = tp
        self.regexp = re.compile(regexp)


class PythonBackendPerfTestConfig:
    """
    Configurations defining the LLM Backend perf test.
    """

    def __init__(self,
                 model_name: str,
                 metrics: Dict[Metric, float],
                 batch_size: int = -1,
                 input_len: int = -1,
                 output_len: int = -1,
                 num_runs: int = 10):
        self.model_name = model_name
        self.metrics = metrics
        self.batch_size = batch_size
        self.input_len = input_len
        self.output_len = output_len
        self.num_runs = num_runs

        assert self.input_len, "invalid input_len"

    def __str__(self):
        entries = [self.model_name, f"bs:{self.batch_size}"]

        if self.output_len:
            entries.append(
                f"input_output_len:{self.input_len},{self.output_len}")
        else:
            entries.append(f"input_len:{self.input_len}")

        entries.append(f"num_runs:{self.num_runs}")
        return "-".join(entries)


class PythonBackendPerfTest(AbstractPerfScriptTestClass):
    """
    Base class for perf tests with multiple metrics.
    """

    def __init__(self, config: PythonBackendPerfTestConfig):
        assert isinstance(config, PythonBackendPerfTestConfig)
        self._config = config
        self._current_metric = None

    def get_test_name(self) -> str:
        return str(self._config)

    def set_runtime_configs(self, benchmark_script, working_dir,
                            perf_cache_fpath) -> None:
        self._benchmark_script = benchmark_script
        self._working_dir = working_dir
        self._perf_cache_fpath = perf_cache_fpath

    def get_commands(self) -> List[PerfScriptTestCmd]:
        benchmark_cmd = [
            self._benchmark_script, f"--start_len={self._config.input_len}",
            f"--output_len={self._config.output_len}",
            f"--batch_size={self._config.batch_size}",
            f"--num_runs={self._config.num_runs}", "--warm_up"
        ]

        benchmark_cmd = PerfScriptTestCmd(benchmark_cmd, isPython=True)

        return [benchmark_cmd]

    def get_perf_result(self, outputs: List[str]) -> float:
        metric = self._current_metric
        # check perf time
        perf_times = [
            metric.regexp.search(line) for line in outputs[0].split("\n")
        ]
        perf_times = [float(match.group(1)) for match in perf_times if match]

        if len(perf_times) == 1:
            return perf_times[0]

        raise RuntimeError("Cannot find perf result from perf script logs!")

    def get_threshold(self) -> float:
        return self._config.metrics[self._current_metric]

    def get_metric_type(self) -> PerfMetricType:
        return self._current_metric.tp

    def run_metrics(self, turtle_case_name, llm_backend_venv, gpu_clock_lock,
                    trt_session_data_writer, output_dir):
        outputs = []
        for metric in self._config.metrics:
            self._current_metric = metric
            fullname = turtle_case_name.replace(
                "test_perf",
                f"test_perf_metric_{self._current_metric.tp.lower()}")
            try:
                outputs = self.run_ex(fullname,
                                      llm_backend_venv,
                                      gpu_clock_lock,
                                      trt_session_data_writer,
                                      output_dir,
                                      outputs=outputs)
            except Exception as e:
                print(f"perf command failed with {e}")


LATENCY = Metric(PerfMetricType.LATENCY, r"Latency: (\d+\.\d+) ms")
BATCH_SIZE = [1, 2, 4, 8, 16, 32, 64]
IN_OUT_SEQ = [(128, 8), (512, 32)]


def generate_llm_perf_config():
    configs = []
    model = "gpt_350m"
    for bs in BATCH_SIZE:
        for input_len, output_len in IN_OUT_SEQ:
            configs.append(
                PythonBackendPerfTestConfig(
                    model,
                    {
                        LATENCY: 0.1,
                    },
                    batch_size=bs,
                    input_len=input_len,
                    output_len=output_len,
                ))
    return configs


# Generate test list and test name list.
LLM_TESTS = [
    PythonBackendPerfTest(config) for config in generate_llm_perf_config()
]


class TestPythonBackendPerf:

    @pytest.mark.parametrize("case",
                             LLM_TESTS,
                             ids=lambda c: c.get_test_name())
    def test_perf(self, setup_gpt_python_backend_perf_test_env,
                  turtle_case_name, trt_performance_cache_fpath,
                  trt_gpu_clock_lock, llm_session_data_writer, output_dir,
                  llm_backend_venv, llm_backend_root, case):
        """
        The actual test definition for TensorRT LLM Backend perf test.
        """
        benchmark_script = os.path.join(llm_backend_root, "tools", "gpt",
                                        "benchmark_core_model.py")
        working_dir = llm_backend_venv.get_working_directory()

        case.set_runtime_configs(benchmark_script, working_dir,
                                 trt_performance_cache_fpath)
        case.run_metrics(turtle_case_name, llm_backend_venv,
                         trt_gpu_clock_lock, llm_session_data_writer,
                         output_dir)


class InflightBatchingPerfTestConfig:
    """
    Configurations defining the LLM Backend perf test.
    """

    def __init__(self,
                 model_name: str,
                 metrics: Dict[Metric, float],
                 concurrency: int = -1,
                 max_input_len: int = -1):
        self.model_name = model_name
        self.metrics = metrics
        self.concurrency = concurrency
        self.max_input_len = max_input_len

    def __str__(self):
        entries = [
            self.model_name, f"concurrency:{self.concurrency}",
            f"max_input_len:{self.max_input_len}"
        ]

        return "-".join(entries)


class InflightBatchingMetricPerfTest(AbstractPerfScriptTestClass):
    """
    Base class for perf tests with multiple metrics.
    """

    def __init__(self, config: InflightBatchingPerfTestConfig):
        assert isinstance(config, InflightBatchingPerfTestConfig)
        self._config = config
        self._current_metric = None

    def get_test_name(self) -> str:
        return str(self._config)

    def set_runtime_configs(self, benchmark_script, working_dir,
                            perf_cache_fpath, dataset, tokenizer_dir,
                            tokenizer_type) -> None:
        self._benchmark_script = benchmark_script
        self._working_dir = working_dir
        self._perf_cache_fpath = perf_cache_fpath
        self._dataset = dataset
        self._tokenizer_dir = tokenizer_dir
        self._tokenizer_type = tokenizer_type

    def get_commands(self) -> List[PerfScriptTestCmd]:
        benchmark_cmd = [
            self._benchmark_script,
            f"--concurrency={self._config.concurrency}",
            f"--max-input-len={self._config.max_input_len}", "dataset",
            f"--dataset={self._dataset}",
            f"--tokenizer-dir={self._tokenizer_dir}",
            f"--tokenizer-type={self._tokenizer_type}"
        ]

        benchmark_cmd = PerfScriptTestCmd(benchmark_cmd, isPython=True)

        return [benchmark_cmd]

    def get_perf_result(self, outputs: List[str]) -> float:
        metric = self._current_metric
        # check perf time
        perf_times = [
            metric.regexp.search(line) for line in outputs[0].split("\n")
        ]
        perf_times = [float(match.group(1)) for match in perf_times if match]

        if len(perf_times) == 1:
            return perf_times[0]

        raise RuntimeError("Cannot find perf result from perf script logs!")

    def get_threshold(self) -> float:
        return self._config.metrics[self._current_metric]

    def get_metric_type(self) -> PerfMetricType:
        return self._current_metric.tp

    def run_metrics(self, turtle_case_name, llm_backend_venv, gpu_clock_lock,
                    trt_session_data_writer, output_dir):
        outputs = []
        for metric in self._config.metrics:
            self._current_metric = metric
            fullname = turtle_case_name.replace(
                "test_perf",
                f"test_perf_metric_{self._current_metric.tp.lower()}")
            try:
                outputs = self.run_ex(fullname,
                                      llm_backend_venv,
                                      gpu_clock_lock,
                                      trt_session_data_writer,
                                      output_dir,
                                      outputs=outputs)
            except Exception as e:
                print(f"perf command failed with {e}")


CONCURRENCY = [2, 4, 8, 16]
MAX_INPUT_LEN = [300, 400, 500]


def generate_llm_ib_perf_config():
    configs = []
    model = "llama_v2_7b"
    for concurrency in CONCURRENCY:
        for max_input_len in MAX_INPUT_LEN:
            configs.append(
                InflightBatchingPerfTestConfig(
                    model,
                    {
                        LATENCY: 0.1,
                    },
                    concurrency=concurrency,
                    max_input_len=max_input_len,
                ))
    return configs


# Generate test list and test name list.
LLM_IB_TESTS = [
    InflightBatchingMetricPerfTest(config)
    for config in generate_llm_ib_perf_config()
]


class TestInflightBatchingPerf:

    @pytest.mark.parametrize("case",
                             LLM_IB_TESTS,
                             ids=lambda c: c.get_test_name())
    def test_perf(self, setup_llama_ifb_perf_test_env, turtle_case_name,
                  trt_performance_cache_fpath, trt_gpu_clock_lock,
                  llm_session_data_writer, output_dir, llm_backend_venv,
                  llm_backend_root, case, llama_v2_tokenizer_model_root):
        """
        The actual test definition for TensorRT LLM Backend perf test.
        """
        benchmark_script = os.path.join(llm_backend_root, "tools",
                                        "inflight_batcher_llm",
                                        "benchmark_core_model.py")
        dataset_path = os.path.join(llm_backend_root, "tools", "dataset",
                                    "mini_cnn_eval.json")
        working_dir = llm_backend_venv.get_working_directory()
        tokenizer_type = "llama"
        case.set_runtime_configs(benchmark_script, working_dir,
                                 trt_performance_cache_fpath, dataset_path,
                                 llama_v2_tokenizer_model_root, tokenizer_type)
        case.run_metrics(turtle_case_name, llm_backend_venv,
                         trt_gpu_clock_lock, llm_session_data_writer,
                         output_dir)
