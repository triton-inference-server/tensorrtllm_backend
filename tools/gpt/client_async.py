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
from datetime import datetime

import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from utils import token_encoder, utils

# GPT3 Related variables
# Reference : https://github.com/NVIDIA/FasterTransformer/blob/main/sample/pytorch/gpt_sample.py
MERGES_FILE = "gpt2-merges.txt"
VOCAB_FILE = "gpt2-vocab.json"

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
    parser.add_argument('-c',
                        '--concurrency',
                        type=int,
                        default=1,
                        required=False,
                        help='Specify concurrency')
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

    encoder = token_encoder.get_encoder(VOCAB_FILE, MERGES_FILE)
    input_start_ids = np.array([encoder.encode(FLAGS.text)], np.int32)
    inputs = utils.prepare_inputs(input_start_ids, FLAGS)

    start_time = datetime.now()

    with utils.create_inference_server_client(FLAGS.protocol,
                                              FLAGS.url,
                                              concurrency=FLAGS.concurrency,
                                              verbose=FLAGS.verbose) as client:
        if FLAGS.protocol == "http":
            async_requests = utils.send_requests_async('tekit',
                                                       inputs,
                                                       client,
                                                       FLAGS,
                                                       request_parallelism=1)
            results = utils.get_http_results(async_requests)
        else:
            user_data = utils.send_requests_async('tekit',
                                                  inputs,
                                                  client,
                                                  FLAGS,
                                                  request_parallelism=1)
            results = utils.get_grpc_results(user_data, request_parallelism=1)
    output_ids = results[0].as_numpy("output_ids")

    stop_time = datetime.now()
    latencies = (stop_time - start_time).total_seconds() * 1000.0
    print(f"[INFO] Latency: {latencies} ms")

    output_ids = output_ids.reshape(
        (output_ids.size, )).tolist()[input_start_ids.shape[1]:]
    output_text = encoder.decode(output_ids)
    print(f'Input: {FLAGS.text}')
    print(f'Output: {output_text}')
