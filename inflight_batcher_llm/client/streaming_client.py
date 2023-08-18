#!/usr/bin/env python
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
import sys
from functools import partial

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

#
# Simple streaming client for TRT-LLM inflight bacthing backend
#
# In order for this code to work properly, config.pbtxt must contain these values:
#
# model_transaction_policy {
#   decoupled: True
# }
#
# parameters: {
#   key: "gpt_model_type"
#   value: {
#     string_value: "inflight_batching"
#   }
# }
#
# In order for gpt_model_type 'inflight_batching' to work, you must copy engine from
#
# tekit/cpp/tests/resources/models/rt_engine/gpt2/fp16-inflight-batching-plugin/1-gpu/
#


class UserData:

    def __init__(self):
        self._completed_requests = queue.Queue()


# Define the callback function. Note the last two parameters should be
# result and error. InferenceServerClient would povide the results of an
# inference as grpcclient.InferResult in result. For successful
# inference, error will be None, otherwise it will be an object of
# tritonclientutils.InferenceServerException holding the error details
def callback(user_data, result, error):
    if error:
        user_data._completed_requests.put(error)
    else:
        print(result)
        user_data._completed_requests.put(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        required=False,
        default=False,
        help="Enable verbose output",
    )
    parser.add_argument(
        "-u",
        "--url",
        type=str,
        required=False,
        default="localhost:8001",
        help="Inference server URL. Default is localhost:8001.",
    )
    parser.add_argument(
        "-s",
        "--ssl",
        action="store_true",
        required=False,
        default=False,
        help="Enable SSL encrypted channel to the server",
    )
    parser.add_argument(
        "-t",
        "--stream-timeout",
        type=float,
        required=False,
        default=None,
        help="Stream timeout in seconds. Default is None.",
    )
    parser.add_argument(
        "-r",
        "--root-certificates",
        type=str,
        required=False,
        default=None,
        help="File holding PEM-encoded root certificates. Default is None.",
    )
    parser.add_argument(
        "-p",
        "--private-key",
        type=str,
        required=False,
        default=None,
        help="File holding PEM-encoded private key. Default is None.",
    )
    parser.add_argument(
        "-x",
        "--certificate-chain",
        type=str,
        required=False,
        default=None,
        help="File holding PEM-encoded certificate chain. Default is None.",
    )
    parser.add_argument(
        "-C",
        "--grpc-compression-algorithm",
        type=str,
        required=False,
        default=None,
        help=
        "The compression algorithm to be used when sending request to server. Default is None.",
    )

    FLAGS = parser.parse_args()

    print('=========')
    input_ids_data = np.array(
        [[28524, 287, 5093, 12, 23316, 4881, 11, 30022, 263, 8776, 355, 257]],
        dtype=np.int32)
    input_lengths_data = np.array([[12]], dtype=np.int32)
    request_output_len_data = np.array([[8]], dtype=np.int32)

    inputs = [
        grpcclient.InferInput('input_ids', [1, 12], "INT32"),
        grpcclient.InferInput('input_lengths', [1, 1], "INT32"),
        grpcclient.InferInput('request_output_len', [1, 1], "INT32"),
    ]
    inputs[0].set_data_from_numpy(input_ids_data)
    inputs[1].set_data_from_numpy(input_lengths_data)
    inputs[2].set_data_from_numpy(request_output_len_data)

    user_data = UserData()
    with grpcclient.InferenceServerClient(
            url=FLAGS.url,
            verbose=FLAGS.verbose,
            ssl=FLAGS.ssl,
            root_certificates=FLAGS.root_certificates,
            private_key=FLAGS.private_key,
            certificate_chain=FLAGS.certificate_chain,
    ) as triton_client:
        try:
            # Establish stream
            triton_client.start_stream(
                callback=partial(callback, user_data),
                stream_timeout=FLAGS.stream_timeout,
            )
            # Send request
            triton_client.async_stream_infer('tensorrt_llm_streaming',
                                             inputs,
                                             request_id="12345",
                                             parameters={'Streaming': 1})
            eos = False
            while not eos:
                result = user_data._completed_requests.get()
                if type(result) == InferenceServerException:
                    print(result)
                    sys.exit(1)
                else:
                    print("*******")
                    print('Response: {}'.format(result.get_response()))
                    output_ids = result.as_numpy('output_ids')
                    print('output_ids= {}'.format(output_ids))
                    if output_ids[0][-1] == 50256:
                        eos = True
        except Exception as e:
            print("channel creation failed: " + str(e))
            sys.exit()
