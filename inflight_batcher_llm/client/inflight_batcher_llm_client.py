#!/usr/bin/env python
# Copyright 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np
import tritonclient.http as httpclient

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-u',
        '--url',
        type=str,
        required=False,
        default='localhost:8000',
        help='Inference server URL. Default is localhost:8000.')

    FLAGS = parser.parse_args()

    # For the HTTP client, need to specify large enough concurrency to
    # issue all the inference requests to the server in parallel. For
    # this example we want to be able to send 2 requests concurrently.
    try:
        concurrent_request_count = 2
        triton_client = httpclient.InferenceServerClient(
            url=FLAGS.url, concurrency=concurrent_request_count)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit(1)

    # First send a single request to the nonbatching model.
    print('=========')
    input_ids_data = np.array(
        [[28524, 287, 5093, 12, 23316, 4881, 11, 30022, 263, 8776, 355, 257]],
        dtype=np.int32)
    input_lengths_data = np.array([[12]], dtype=np.int32)
    request_output_len_data = np.array([[8]], dtype=np.int32)

    inputs = [
        httpclient.InferInput('input_ids', [1, 12], "INT32"),
        httpclient.InferInput('input_lengths', [1, 1], "INT32"),
        httpclient.InferInput('request_output_len', [1, 1], "INT32"),
    ]
    inputs[0].set_data_from_numpy(input_ids_data)
    inputs[1].set_data_from_numpy(input_lengths_data)
    inputs[2].set_data_from_numpy(request_output_len_data)

    # you can pass parameters with each inference request
    params = {"parameter1": "parameter1", "parameter2": 2}
    # sequence_id in client API equals correlation_id on server side
    result = triton_client.infer('tensorrt_llm',
                                 inputs,
                                 sequence_id=12345,
                                 sequence_start=True,
                                 sequence_end=True,
                                 parameters=params)

    print('Response: {}'.format(result.get_response()))
    print('output_ids= {}'.format(result.as_numpy('output_ids')))
