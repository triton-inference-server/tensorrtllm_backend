import abc
import os
import re
from datetime import datetime
from enum import Enum
from typing import List, NamedTuple

from trt_test.gpu_clock_lock import (GPUClockLock,
                                     InvalidGPUMonitoringResultError)
from trt_test.misc import check_output, temp_wd
from trt_test.perf.data_export import (GPU_MONITORING_FORMAT_KEYS, write_csv,
                                       write_gpu_monitoring_no_test_results,
                                       write_yaml)
from trt_test.session_data_writer import SessionDataWriter
from trt_test.venv_runner import VirtualenvRunner


class PerfMetricType(str, Enum):
    """
    An string-enum type to define what kind of perf metric it is. While it is not used by TURTLE, it is used by QA to
    set up special threshold criteria for each type of metrics (like >50MB for engine size increase, etc.).
    """
    INFERENCE_TIME = "INFERENCE_TIME"
    BUILD_TIME = "BUILD_TIME"
    PEAK_CPU_MEMORY = "PEAK_CPU_MEMORY"
    PEAK_GPU_MEMORY = "PEAK_GPU_MEMORY"
    ENGINE_SIZE = "ENGINE_SIZE"
    CONTEXT_GPU_MEMORY = "CONTEXT_GPU_MEMORY"
    LATENCY = "LATENCY"
    TOKENS_PER_SEC = "TOKENS_PER_SEC"


class PerfScriptTestCmd(NamedTuple):
    cmd: List[str]
    isPython: bool


class AbstractPerfScriptTestClass(abc.ABC):
    """
    Abstract class for all script-based perf tests.
    """

    @abc.abstractmethod
    def get_test_name(self) -> str:
        """
        Define the test name for this test, which will appear in the "[...]" part of the generate test names.
        WARNING: Please keep backward compatibility in get_test_name() method when adding new tests!
        Changing test names means we will lose test history in NvRegress and in PerfDB!
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def set_runtime_configs(self, *args) -> None:
        """
        Set the runtime configs (like directory paths, compute capability, etc.) for the test.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_commands(self) -> List[PerfScriptTestCmd]:
        """
        Get the commands to run the test. Should return a list of tuple.
        The 1st item cmd in tuple is command, where each command is a list of args.
        The 2nd item isPython in tuple is a bool which indicates whether this is a python script.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_perf_result(self, outputs: List[str]) -> float:
        """
        Get the perf result (latency) from the output logs of each command.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_threshold(self) -> float:
        """
        Get the relative threshold used to flag a perf regression compared to perf baseline.
        """
        raise NotImplementedError()

    def get_absolute_threshold(self) -> float:
        """
        Get the absolute threshold used to flag a perf regression compared to perf baseline.
        Perf comparison will only fail if it exceeds both relative and absolute thresholds.
        Note: This is not honored by TURTLE for now, but we can add the support later.
        """
        return 0.0

    def get_metric_type(self) -> PerfMetricType:
        """
        Get the type of perf metric. This does not affect TURTLE for now, but QA uses this field to set up special
        threshold criteria depending on the metric type.
        """
        return PerfMetricType.INFERENCE_TIME

    def get_working_dir(self) -> str:
        """
        Get the working directory to run the commands in. Default is the current working directory.
        Derived classes can override this function if a different working directory is needed.
        """
        return os.getcwd()

    def run_ex(self,
               full_test_name: str,
               venv: VirtualenvRunner,
               trt_gpu_clock_lock: GPUClockLock,
               trt_session_data_writer: SessionDataWriter,
               output_dir: str,
               trt_run=None,
               outputs=[]) -> List[str]:
        """
        Run the commands and write the results to the output csv and/or yaml files.
        """

        # Get the commands.
        commands = self.get_commands()
        outputs = outputs.copy()  # avoid modifying argument directly

        # Initialize result status.
        self._perf_result = None
        self._result_state = "valid"
        self._error = None
        self._gpu_clock_lock = trt_gpu_clock_lock

        if len(outputs) and len(outputs) != len(commands):
            raise RuntimeError("Cached output logs do not match commands")

        # Start the timer.
        self._start_timestamp = datetime.utcnow()
        try:
            # Lock GPU clock and start monitoring.
            if not len(outputs):
                with self._gpu_clock_lock:
                    with temp_wd(self.get_working_dir()):
                        # Run the test commands.
                        for command in commands:
                            if command.isPython:
                                output = venv.run_cmd(command.cmd,
                                                      caller=check_output)
                                print(output)
                            else:
                                assert trt_run, "The trt_run runner is not provided!"
                                output = ""

                                def append_to_output(line):
                                    nonlocal output
                                    output += line
                                    line = line.rstrip()
                                    if line != "":
                                        print(line)

                                trt_run.run_ex(command.cmd,
                                               write_func=append_to_output)
                            outputs.append(output)

        except InvalidGPUMonitoringResultError:
            # Mark result state as invalid when GPU monitoring result is invalid.
            self._result_state = "invalid"

        except Exception as e:
            # Mark result state as failed if anything else went wrong.
            self._result_state = "failed"
            self._error = e

        finally:
            # Always parse the perf result from the test outputs.
            self._perf_result = self.get_perf_result(outputs)

            # Stop the timer
            self._end_timestamp = datetime.utcnow()

            # Write results to output csv and/or yaml files.
            self._write_result(full_test_name, trt_session_data_writer,
                               output_dir, outputs)

            # Raise the error if anything went wrong.
            if self._error is not None:
                raise RuntimeError("Test failed. Error: {:}".format(
                    self._error))

            return outputs

    def _write_result(self, full_test_name: str,
                      trt_session_data_writer: SessionDataWriter,
                      output_dir: str, outputs: List[str]) -> None:
        """
        Write the test results and GPU monitoring data to the output csv and/or yaml files.
        """

        # Construct the result dict to write. The keys follow the convention of TRTPerfExecutableRunner.
        gpu_clocks = self._gpu_clock_lock.get_target_gpu_clocks()
        # Remove the prefix, which includes the platform info, for network_hash.
        short_test_name = full_test_name.split("::")[-1]
        test_description_dict = {
            "network_name":
            self.get_test_name(),
            "network_hash":
            short_test_name,  # This is used by the PerfDB to identify a test.
            "sm_clk":
            gpu_clocks["sm_clk"],
            "mem_clk":
            gpu_clocks["mem_clk"],
            "gpu_idx":
            self._gpu_clock_lock.get_gpu_properties().get(
                "index", self._gpu_clock_lock.get_gpu_id()),
        }
        test_result_dict = {
            "turtle_case_name":
            full_test_name,
            "test_name":
            short_test_name,
            "raw_result":
            "\n".join(outputs),
            "perf_metric":
            self._perf_result,
            "total_time__sec":
            (self._end_timestamp - self._start_timestamp).total_seconds(),
            "start_timestamp":
            self._start_timestamp.strftime("%Y-%m-%d %H:%M:%S %z").rstrip(),
            "end_timestamp":
            self._end_timestamp.strftime("%Y-%m-%d %H:%M:%S %z").rstrip(),
            "state":
            self._result_state,
            "command":
            " ; ".join([(("python3 " if command.isPython else "") +
                         " ".join(command.cmd))
                        for command in self.get_commands()]),
            "threshold":
            self.get_threshold(),
            "absolute_threshold":
            self.get_absolute_threshold(),
            "metric_type":
            self.get_metric_type().value,
        }

        # Get GPU monitoring data.
        self._gpu_monitor_data = self._gpu_clock_lock.get_state_data()
        if self._gpu_monitor_data is None:
            print("WARNING: No GPU monitoring data!")

        # Write results in CSV format.
        if "csv" in trt_session_data_writer._output_formats:
            csv_name = "perf_script_test_results.csv"
            cvs_result_dict = {**test_description_dict, **test_result_dict}
            cvs_result_dict["raw_result"] = cvs_result_dict[
                "raw_result"].replace("\n", "\\n")
            write_csv(output_dir,
                      csv_name, [cvs_result_dict],
                      list(cvs_result_dict.keys()),
                      append_mode=os.path.exists(
                          os.path.join(output_dir, csv_name)))
            if self._gpu_monitor_data is not None:
                write_gpu_monitoring_no_test_results(output_dir,
                                                     self._gpu_monitor_data,
                                                     output="csv",
                                                     append_mode=True)

        # Write results in YAML format.
        if "yaml" in trt_session_data_writer._output_formats:
            yaml_name = "perf_script_test_results.yaml"
            monitor_data_dict = [{
                key: getattr(i, key)
                for key in GPU_MONITORING_FORMAT_KEYS
            } for i in self._gpu_monitor_data]
            yaml_result_dict = {
                "monitor_data": {
                    "cpu": [],
                    "os": [],
                    "gpu": monitor_data_dict,
                },
                "test_description": test_description_dict,
                "test_result": test_result_dict,
            }
            yaml_result_dict = {
                yaml_result_dict["test_result"]["test_name"]: yaml_result_dict
            }
            write_yaml(output_dir,
                       yaml_name,
                       yaml_result_dict,
                       append_mode=os.path.exists(
                           os.path.join(output_dir, yaml_name)))

    def match_string(self, perf_logs: List[str], match: str) -> str:
        if len(perf_logs) != 1:
            raise RuntimeError(
                f"Incorrect number of perf logs: {len(perf_logs)} != 1")

        perf_log = perf_logs[0].split('\n')
        regex = re.compile(match)
        for line in perf_log:
            builder_perf_result_match = regex.search(line)
            if builder_perf_result_match:
                return builder_perf_result_match.group(1)
        return None
