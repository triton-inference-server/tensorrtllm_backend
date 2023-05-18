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
from builtins import range
from datetime import datetime
from functools import partial

import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype
from utils import token_encoder

# GPT3 Related variables
# Reference : https://github.com/NVIDIA/FasterTransformer/blob/main/sample/pytorch/gpt_sample.py
MERGES_FILE = "gpt2-merges.txt"
VOCAB_FILE = "gpt2-vocab.json"

START_ID = 50256
END_ID = 50256


class UserData:

    def __init__(self):
        self._completed_requests = queue.Queue()


# Callback function used for async_stream_infer()
def completion_callback(user_data, result, error):
    # passing error raise and handling out
    user_data._completed_requests.put((result, error))


def prepare_tensor(name, input, protocol):
    client_util = httpclient if protocol == "http" else grpcclient
    t = client_util.InferInput(name, input.shape,
                               np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t


def prepare_inputs(input_start_ids, flags):
    output_len = np.ones([input_start_ids.shape[0], 1]).astype(
        np.uint32) * FLAGS.output_len
    runtime_top_k = (flags.topk *
                     np.ones([input_start_ids.shape[0], 1])).astype(np.uint32)
    runtime_top_p = flags.topp * \
        np.ones([input_start_ids.shape[0], 1]).astype(np.float32)
    beam_search_diversity_rate = 0.0 * \
        np.ones([input_start_ids.shape[0], 1]).astype(np.float32)
    temperature = 1.0 * \
        np.ones([input_start_ids.shape[0], 1]).astype(np.float32)
    len_penalty = 1.0 * \
        np.ones([input_start_ids.shape[0], 1]).astype(np.float32)
    repetition_penalty = 1.0 * \
        np.ones([input_start_ids.shape[0], 1]).astype(np.float32)
    random_seed = 0 * \
        np.ones([input_start_ids.shape[0], 1]).astype(np.uint64)
    is_return_log_probs = True * \
        np.ones([input_start_ids.shape[0], 1]).astype(bool)
    beam_width = (flags.beam_width *
                  np.ones([input_start_ids.shape[0], 1])).astype(np.uint32)
    start_ids = 50256 * \
        np.ones([input_start_ids.shape[0], 1]).astype(np.uint32)
    end_ids = 50256 * \
        np.ones([input_start_ids.shape[0], 1]).astype(np.uint32)
    bad_words_list = np.concatenate([
        np.zeros([input_start_ids.shape[0], 1, 1]).astype(np.int32),
        (-1 * np.ones([input_start_ids.shape[0], 1, 1])).astype(np.int32)
    ],
                                    axis=1)
    stop_word_list = np.concatenate([
        np.zeros([input_start_ids.shape[0], 1, 1]).astype(np.int32),
        (-1 * np.ones([input_start_ids.shape[0], 1, 1])).astype(np.int32)
    ],
                                    axis=1)
    inputs = [
        prepare_tensor("input_ids", input_start_ids, flags.protocol),
        # prepare_tensor("input_lengths", input_len, flags.protocol),
        prepare_tensor("request_output_len", output_len, flags.protocol),
        prepare_tensor("runtime_top_k", runtime_top_k, flags.protocol),
        prepare_tensor("runtime_top_p", runtime_top_p, flags.protocol),
        prepare_tensor("beam_search_diversity_rate",
                       beam_search_diversity_rate, flags.protocol),
        prepare_tensor("temperature", temperature, flags.protocol),
        prepare_tensor("len_penalty", len_penalty, flags.protocol),
        prepare_tensor("repetition_penalty", repetition_penalty,
                       flags.protocol),
        prepare_tensor("random_seed", random_seed, flags.protocol),
        # prepare_tensor("is_return_log_probs", is_return_log_probs, flags.protocol),
        prepare_tensor("beam_width", beam_width, flags.protocol),
        # prepare_tensor("start_id", start_ids, flags.protocol),
        # prepare_tensor("end_id", end_ids, flags.protocol),
        # prepare_tensor("bad_words_list", bad_words_list, flags.protocol),
        # prepare_tensor("stop_words_list", stop_word_list, flags.protocol),
    ]
    return inputs


def send_http_requests(client, inputs, request_parallelism):
    model_name = "tekit"
    async_requests = []
    for _ in range(request_parallelism):
        async_requests.append(client.async_infer(model_name, inputs))

    results = []
    for async_request in async_requests:
        results.append(async_request.get_result())
    return results


def send_grpc_requests(client, inputs, request_parallelism):
    model_name = "tekit"
    async_requests = []
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
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        help='Inference server URL.')
    parser.add_argument(
        '-i',
        '--protocol',
        type=str,
        required=False,
        default='http',
        help='Protocol ("http"/"grpc") used to ' +
        'communicate with inference service. Default is "http".')
    parser.add_argument(
        '-t',
        '--text',
        type=str,
        required=False,
        default='Born in north-east France, Soyer trained as a',
        help='Input text')
    parser.add_argument('-beam',
                        '--beam_width',
                        type=int,
                        default=1,
                        required=False,
                        help='Specify beam width')
    parser.add_argument('-topk',
                        '--topk',
                        type=int,
                        default=1,
                        required=False,
                        help='topk for sampling')
    parser.add_argument('-topp',
                        '--topp',
                        type=float,
                        default=0.0,
                        required=False,
                        help='topp for sampling')
    parser.add_argument('-o',
                        '--output_len',
                        type=int,
                        default=10,
                        required=False,
                        help='Specify output length')

    FLAGS = parser.parse_args()
    if (FLAGS.protocol != "http") and (FLAGS.protocol != "grpc"):
        print(
            "unexpected protocol \"{}\", expects \"http\" or \"grpc\"".format(
                FLAGS.protocol))
        exit(1)

    client_util = httpclient if FLAGS.protocol == "http" else grpcclient
    if FLAGS.url is None:
        FLAGS.url = "localhost:8000" if FLAGS.protocol == "http" else "localhost:8001"

    concurrency = 20
    if FLAGS.protocol == "http":
        client = client_util.InferenceServerClient(FLAGS.url,
                                                   concurrency=concurrency,
                                                   verbose=FLAGS.verbose)
    else:
        client = client_util.InferenceServerClient(FLAGS.url,
                                                   verbose=FLAGS.verbose)

    encoder = token_encoder.get_encoder(VOCAB_FILE, MERGES_FILE)
    input_start_ids = np.array([encoder.encode(FLAGS.text)], np.int32)
    inputs = prepare_inputs(input_start_ids, FLAGS)

    request_parallelism = 1
    start_time = datetime.now()
    if FLAGS.protocol == "http":
        results = send_http_requests(client, inputs, request_parallelism)
    else:
        results = send_grpc_requests(client, inputs, request_parallelism)
    stop_time = datetime.now()
    latencies = (stop_time - start_time).total_seconds() * 1000.0
    print(f"[INFO] Latency: {latencies} ms")

    for result in results:
        output_ids = result.as_numpy('output_ids')
        output_ids = output_ids.reshape(
            (output_ids.size, )).tolist()[input_start_ids.shape[1]:]
        output_text = encoder.decode(output_ids)
        print(f'Input: {FLAGS.text}')
        print(f'Output: {output_text}')
