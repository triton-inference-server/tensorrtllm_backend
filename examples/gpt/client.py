#!/usr/bin/python

# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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

import argparse
import queue
import statistics as s
from builtins import range
from datetime import datetime
from functools import partial

import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype


class UserData:

    def __init__(self):
        self._completed_requests = queue.Queue()


# Callback function used for async_stream_infer()
def completion_callback(user_data, result, error):
    # passing error raise and handling out
    user_data._completed_requests.put((result, error))


def prepare_tensor(name, input):
    t = client_util.InferInput(name, input.shape,
                               np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t


def send_http_requests(client, input_start_ids, request_parallelism):
    model_name = "gpt"
    async_requests = []
    inputs = [prepare_tensor("input_ids", input_start_ids)]
    for _ in range(request_parallelism):
        async_requests.append(client.async_infer(model_name, inputs))

    results = []
    for async_request in async_requests:
        results.append(async_request.get_result())
    return results


def send_grpc_requests(client, input_start_ids, request_parallelism):
    model_name = "gpt"
    async_requests = []
    inputs = [prepare_tensor("input_ids", input_start_ids)]
    user_data = UserData()
    for _ in range(request_parallelism):
        async_requests.append(
            client.async_infer(model_name, inputs,
                               partial(completion_callback, user_data)))

    results = []
    processed_count = 0
    while processed_count < request_parallelism:
        (results, error) = user_data._completed_requests.get()
        processed_count += 1
        if error is not None:
            raise RuntimeError(error)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument(
        '-i',
        '--protocol',
        type=str,
        required=False,
        default='http',
        help='Protocol ("http"/"grpc") used to ' +
        'communicate with inference service. Default is "http".')
    parser.add_argument('-w',
                        '--warm_up',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable warm_up before benchmark')
    parser.add_argument(
        '-n',
        '--num_runs',
        type=int,
        default=1,
        required=False,
        help="Spedifty number of runs to get the average latency")

    FLAGS = parser.parse_args()
    if (FLAGS.protocol != "http") and (FLAGS.protocol != "grpc"):
        print(
            "unexpected protocol \"{}\", expects \"http\" or \"grpc\"".format(
                FLAGS.protocol))
        exit(1)

    client_util = httpclient if FLAGS.protocol == "http" else grpcclient
    concurrency = 20
    request_parallelism = 10
    url = 'localhost:8000' if FLAGS.protocol == "http" else 'localhost:8001'
    input_start_ids = np.random.randint(low=0,
                                        high=50000,
                                        size=(4, 128),
                                        dtype=np.int32)
    if FLAGS.protocol == "http":
        client = client_util.InferenceServerClient(url,
                                                   concurrency=concurrency,
                                                   verbose=FLAGS.verbose)
    else:
        client = client_util.InferenceServerClient(url, verbose=FLAGS.verbose)

    latencies = []
    for i in range(FLAGS.num_runs):
        start_time = datetime.now()
        if FLAGS.protocol == "http":
            send_http_requests(client, input_start_ids, request_parallelism)
        else:
            send_grpc_requests(client, input_start_ids, request_parallelism)
        stop_time = datetime.now()
        latencies.append((stop_time - start_time).total_seconds() * 1000.0 /
                         request_parallelism)
    if FLAGS.num_runs > 1:
        print(latencies)
        print(f"[INFO] execution time: {s.mean(latencies)} ms")
    else:
        print(f"[INFO] execution time: {latencies[0]} ms")
