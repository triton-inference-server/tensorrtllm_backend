#!/usr/bin/python
# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import json
import os
import sys
from collections import defaultdict

import numpy as np
import requests

BACKEND_ROOT = os.environ.get('BACKEND_ROOT',
                              "/opt/tritonserver/tensorrtllm_backend")
sys.path.append(os.path.join(BACKEND_ROOT, "tools/utils"))
import unittest

import utils

# This unit test was generated because the Triton team needed a
# static test in which an equal number of inferences were distributed
# across the 3 models orchestrated by the ensemble. This is so we could
# compare inference request counts and latencies in an equal environment.
# Many of the tests provided by the TRT team unevenly distribute requests
# so when we poll metrics the tensorrt_llm model, for example, will have
# performed 72 inferences whereas the pre/post models will have only
# performed 49. Further, because of this unequal distribution of requests
# we cannot check whether the latency across the 3 models is <= to the
# latency of the ensemble.

# Consider removing this unit test when the TRT tests have stabilized.


class TRTLLMBaseMetricsTest(unittest.TestCase):

    def _get_metrics(self):
        metrics_url = "http://localhost:8002/metrics"
        r = requests.get(metrics_url)
        r.raise_for_status()
        return r.text

    def _run_infer(self, client, prompts, output_lens):
        model_name = "ensemble"
        async_requests = []
        for i, prompt in enumerate(prompts):
            input0 = [[prompt]]
            input0_data = np.array(input0).astype(object)
            output0_len = np.ones_like(input0).astype(
                np.int32) * output_lens[i]
            bad_words_list = np.array([[""]], dtype=object)
            stop_words_list = np.array([[""]], dtype=object)

            inputs = [
                utils.prepare_tensor("text_input", input0_data, "http"),
                utils.prepare_tensor("max_tokens", output0_len, "http"),
                utils.prepare_tensor("bad_words", bad_words_list, "http"),
                utils.prepare_tensor("stop_words", stop_words_list, "http"),
            ]
            # Request minimal outputs
            outputs = utils.prepare_outputs("http")

            async_requests.append(
                client.async_infer(model_name,
                                   inputs,
                                   outputs=outputs,
                                   request_id=str(i)))

        try:
            utils.get_http_results(async_requests)
        except Exception as e:
            print("Failed receiving responses: " + str(e))
            sys.exit(1)

    def _all_equal(self, iterable):
        return all(item == iterable[0] for item in iterable)

    def _verify_base_metrics(self, filename):
        # FIXME: Custom parsing is messy. As part of the Triton
        # CLI work, we should add a metrics client API that will
        # return the metrics in a neatly formatted JSON.
        model_metrics = defaultdict(dict)
        with open(filename) as metrics_file:
            for line in metrics_file:
                if line[0] != "#" and "nv_inference" in line:
                    # Splits metric line into:
                    # ex. 'nv_inference_request_success', '{model="ensemble",version="1"}', '104'
                    model_data = line.replace("{", " {").split()
                    key = model_data[0].replace("nv_inference_", "")
                    model = model_data[1].split('"')[1]
                    value = model_data[2]
                    model_metrics[model][key] = value

        print(json.dumps(model_metrics, indent=4))

        # Assert the expected models are in the metrics output
        expected_models = [
            "ensemble", "preprocessing", "postprocessing", "tensorrt_llm"
        ]
        self.assertTrue(
            all(model in model_metrics for model in expected_models))

        # Assert each model records the same number of metrics
        self.assertTrue(
            self._all_equal(
                [len(model_metrics[model].keys()) for model in model_metrics]))

        # Assert models have the same counts
        count_keys = [
            "request_success", "request_failure", "count", "exec_count",
            "pending_request_count"
        ]
        for stat in count_keys:
            if stat == "exec_count":
                # Dynamic batching is enabled for the post-processing model, so
                # the 'exec_count' will not be the same between the
                # postprocessing model and other models.
                self.assertTrue(
                    self._all_equal([
                        model_metrics[model][stat] for model in model_metrics
                        if model != "postprocessing"
                    ]))
            else:
                self.assertTrue(
                    self._all_equal([
                        model_metrics[model][stat] for model in model_metrics
                    ]))

        duration_keys = [
            "request_duration_us", "compute_input_duration_us",
            "compute_infer_duration_us", "compute_output_duration_us"
        ]
        for stat in duration_keys:
            composing_stat_duration = sum([
                int(model_metrics[model][stat]) for model in model_metrics
                if model != "ensemble"
            ])
            ensemble_stat_duration = int(model_metrics["ensemble"][stat])
            self.assertTrue(composing_stat_duration > 0)
            self.assertTrue(ensemble_stat_duration > 0)

    def test_end_to_end(self):
        try:
            client = utils.create_inference_server_client("http",
                                                          "localhost:8000",
                                                          concurrency=128,
                                                          verbose=True)
        except Exception as e:
            print("channel creation failed: " + str(e))
            sys.exit(1)

        max_input_len = 500
        op_tokens_per_word = 1.3
        dataset = "./simple_data.json"

        prompts = []
        output_lens = []
        with open(dataset, "r") as f:
            data_dict = json.load(f)
            for req in data_dict:
                prompt = req["input"] + " " + req["instruction"]
                output = req["output"]
                # 1.3 is a magic number that converts number of words to number of tokens
                if int(len(prompt.split(" ")) /
                       op_tokens_per_word) > max_input_len:
                    continue
                prompts.append(prompt)
                output_lens.append(
                    int(len(output.split(" ")) * op_tokens_per_word))

        self._run_infer(client, prompts, output_lens)
        metrics = self._get_metrics()
        filename = "./base_metrics.out"
        with open(filename, "w+") as metrics_file:
            metrics_file.write(metrics)
        self._verify_base_metrics(filename)


if __name__ == "__main__":
    unittest.main()
