#!/usr/bin/env python
# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import csv
import os
import queue
import sys
import time
from functools import partial

import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from transformers import AutoTokenizer, LlamaTokenizer, T5Tokenizer
from tritonclient.utils import InferenceServerException, np_to_triton_dtype

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
# tensorrt_llm/cpp/tests/resources/models/rt_engine/gpt2/fp16-inflight-batching-plugin/1-gpu/
#

np_bfloat16 = np.dtype('V2', metadata={"dtype": "bfloat16"})

_str_to_np_dict = dict(
    float16=np.float16,
    float32=np.float32,
    int32=np.int32,
    bfloat16=np_bfloat16,
)


def str_dtype_to_np(dtype):
    ret = _str_to_np_dict.get(dtype)
    assert ret is not None, f'Unsupported dtype: {dtype}'
    return ret


class UserData:

    def __init__(self):
        self._completed_requests = queue.Queue()


def prepare_tensor(name, input, protocol):
    client_util = httpclient if protocol == "http" else grpcclient
    t = client_util.InferInput(name, input.shape,
                               np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t


def prepare_inputs(input_ids_data, input_lengths_data, request_output_len_data,
                   beam_width_data, temperature_data, streaming_data, end_id,
                   pad_id, prompt_embedding_table_data,
                   prompt_vocab_size_data):
    protocol = 'grpc'
    inputs = [
        prepare_tensor("input_ids", input_ids_data, protocol),
        prepare_tensor("input_lengths", input_lengths_data, protocol),
        prepare_tensor("request_output_len", request_output_len_data,
                       protocol),
        prepare_tensor("beam_width", beam_width_data, protocol),
        prepare_tensor("temperature", temperature_data, protocol),
        prepare_tensor("streaming", streaming_data, protocol),
        prepare_tensor("end_id", end_id, protocol),
        prepare_tensor("pad_id", pad_id, protocol),
    ]
    if prompt_embedding_table_data is not None:
        inputs += [
            prepare_tensor("prompt_embedding_table",
                           prompt_embedding_table_data, protocol),
            prepare_tensor("prompt_vocab_size", prompt_vocab_size_data,
                           protocol)
        ]

    return inputs


def prepare_stop_signals():

    inputs = [
        grpcclient.InferInput('input_ids', [1, 1], "INT32"),
        grpcclient.InferInput('input_lengths', [1, 1], "INT32"),
        grpcclient.InferInput('request_output_len', [1, 1], "UINT32"),
        grpcclient.InferInput('stop', [1, 1], "BOOL"),
    ]

    inputs[0].set_data_from_numpy(np.empty([1, 1], dtype=np.int32))
    inputs[1].set_data_from_numpy(np.zeros([1, 1], dtype=np.int32))
    inputs[2].set_data_from_numpy(np.array([[0]], dtype=np.uint32))
    inputs[3].set_data_from_numpy(np.array([[True]], dtype='bool'))

    return inputs


# Define the callback function. Note the last two parameters should be
# result and error. InferenceServerClient would povide the results of an
# inference as grpcclient.InferResult in result. For successful
# inference, error will be None, otherwise it will be an object of
# tritonclientutils.InferenceServerException holding the error details
def callback(user_data, result, error):
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)
        if (FLAGS.streaming):
            output_ids = result.as_numpy('output_ids')
            if output_ids != None:
                tokens = list(output_ids[0][0])
                print(tokens, flush=True)


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
        '--text',
        type=str,
        required=False,
        default='Born in north-east France, Soyer trained as a',
        help='Input text')

    parser.add_argument('--input_tokens_csv',
                        type=str,
                        required=False,
                        default='',
                        help='Path to csv file containing the input tokens')

    parser.add_argument(
        '--output_tokens_csv',
        type=str,
        required=False,
        default='',
        help='Path to csv file containing the expected output tokens')

    parser.add_argument(
        '--end_id',
        type=int,
        required=False,
        default=50256,
        help='The token id for end token. Only needed if tokenizer is not used.'
    )

    parser.add_argument(
        '--pad_id',
        type=int,
        required=False,
        default=50256,
        help='The token id for pad token. Only needed if tokenizer is not used.'
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
    parser.add_argument(
        "-S",
        "--streaming",
        action="store_true",
        required=False,
        default=False,
        help="Enable streaming mode. Default is False.",
    )
    parser.add_argument(
        "-c",
        "--check-output",
        action="store_true",
        required=False,
        default=False,
        help="Enable check of output ids for CI",
    )

    parser.add_argument(
        "-b",
        "--beam-width",
        required=False,
        type=int,
        default=1,
        help="Beam width value",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        required=False,
        default=1.0,
        help="temperature value",
    )
    parser.add_argument(
        "--request-output-len",
        type=int,
        required=False,
        default=16,
        help="temperature value",
    )
    parser.add_argument(
        '--stop-after-ms',
        type=int,
        required=False,
        default=0,
        help='Early stop the generation after a few milliseconds')
    parser.add_argument('--tokenizer_dir',
                        type=str,
                        required=False,
                        default='',
                        help='Specify tokenizer directory')
    parser.add_argument('--tokenizer_type',
                        type=str,
                        default='auto',
                        required=False,
                        choices=['auto', 't5', 'llama'],
                        help='Specify tokenizer type')
    parser.add_argument('--request_id',
                        type=str,
                        default='1',
                        required=False,
                        help='The request_id for the stop request')

    parser.add_argument('--prompt_embedding_table_path',
                        type=str,
                        default='',
                        required=False,
                        help='The prompt embedding table to use for ptuning')
    parser.add_argument(
        "--exclude-input-in-output",
        action="store_true",
        required=False,
        default=False,
        help="Expect that output IDs do not contain input IDs",
    )

    parser.add_argument(
        '--prompt_task_id',
        type=int,
        default=0,
        required=False,
        help='The prompt task id in the prompt embedding table')

    parser.add_argument('--dtype',
                        type=str,
                        default='float16',
                        choices=['float16', 'float32', 'bfloat16'])

    FLAGS = parser.parse_args()

    tokenizer = None
    if FLAGS.input_tokens_csv != "":
        with open(FLAGS.input_tokens_csv) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            for row in csv_reader:
                input_ids = [[int(val) for val in row]]
                break
            print(input_ids)

        end_id = FLAGS.end_id
        pad_id = FLAGS.pad_id

    else:
        print('=========')
        if not os.path.exists(FLAGS.tokenizer_dir):
            print(
                "Input tokens are not provided and tokenizer directory does not exist: ",
                FLAGS.tokenizer_dir)
            exit()

        if FLAGS.tokenizer_type == 't5':
            tokenizer = T5Tokenizer(vocab_file=FLAGS.tokenizer_dir,
                                    padding_side='left')
        elif FLAGS.tokenizer_type == 'auto':
            tokenizer = AutoTokenizer.from_pretrained(FLAGS.tokenizer_dir,
                                                      padding_side='left')
        elif FLAGS.tokenizer_type == 'llama':
            tokenizer = LlamaTokenizer.from_pretrained(FLAGS.tokenizer_dir,
                                                       legacy=False,
                                                       padding_side='left')
        else:
            raise AttributeError(
                f'Unexpected tokenizer type: {FLAGS.tokenizer_type}')

        tokenizer.pad_token = tokenizer.eos_token
        pad_id = tokenizer.encode(tokenizer.pad_token,
                                  add_special_tokens=False)[0]
        end_id = tokenizer.encode(tokenizer.eos_token,
                                  add_special_tokens=False)[0]

        input_ids = [tokenizer.encode(FLAGS.text)]
        print(input_ids)

    end_id_data = np.array([[end_id]], dtype=np.uint32)
    pad_id_data = np.array([[pad_id]], dtype=np.uint32)

    #Get the prompt embedding table for the task id
    prompt_embedding_table_data = None
    prompt_vocab_size_data = None
    if (FLAGS.prompt_embedding_table_path != ""):
        prompt_table = np.load(FLAGS.prompt_embedding_table_path)
        prompt_table = prompt_table.astype(str_dtype_to_np(FLAGS.dtype))
        task_vocab_size = prompt_table.shape[1]

        # squeeze the first 2 dimensions
        prompt_embedding_table_data = prompt_table[FLAGS.prompt_task_id]
        prompt_embedding_table_data = np.expand_dims(
            prompt_table[FLAGS.prompt_task_id], axis=0)

        prompt_vocab_size = [[task_vocab_size]]
        prompt_vocab_size_data = np.array(prompt_vocab_size, dtype=np.uint32)

    input_ids_data = np.array(input_ids, dtype=np.int32)
    input_lengths = [[len(ii)] for ii in input_ids]
    input_lengths_data = np.array(input_lengths, dtype=np.int32)
    request_output_len = [[FLAGS.request_output_len]]
    request_output_len_data = np.array(request_output_len, dtype=np.uint32)
    beam_width = [[FLAGS.beam_width]]
    beam_width_data = np.array(beam_width, dtype=np.uint32)
    temperature = [[FLAGS.temperature]]
    temperature_data = np.array(temperature, dtype=np.float32)
    streaming = [[FLAGS.streaming]]
    streaming_data = np.array(streaming, dtype=bool)

    inputs = prepare_inputs(input_ids_data, input_lengths_data,
                            request_output_len_data, beam_width_data,
                            temperature_data, streaming_data, end_id_data,
                            pad_id_data, prompt_embedding_table_data,
                            prompt_vocab_size_data)

    if FLAGS.stop_after_ms > 0:
        stop_inputs = prepare_stop_signals()
    else:
        stop_inputs = None

    request_id = FLAGS.request_id

    if FLAGS.output_tokens_csv != "":
        with open(FLAGS.output_tokens_csv) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            for row in csv_reader:
                expected_output_ids = [int(val) for val in row]
                break
    else:
        expected_output_ids = ([] if FLAGS.exclude_input_in_output else
                               input_ids[0]) + [
                                   21221, 290, 257, 4255, 379, 262, 1957, 7072,
                                   11, 4689, 347, 2852, 2564, 494, 13, 679
                               ]

    if FLAGS.streaming:
        actual_output_ids = [
            [] if FLAGS.exclude_input_in_output else input_ids[0]
        ]
    else:
        actual_output_ids = []

    sequence_lengths = []

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

            if FLAGS.streaming:

                # Establish stream
                triton_client.start_stream(
                    callback=partial(callback, user_data),
                    stream_timeout=FLAGS.stream_timeout,
                )
                # Send request
                triton_client.async_stream_infer(
                    'tensorrt_llm',
                    inputs,
                    request_id=request_id,
                )

                if stop_inputs is not None:

                    time.sleep(FLAGS.stop_after_ms / 1000.0)

                    triton_client.async_stream_infer(
                        'tensorrt_llm',
                        stop_inputs,
                        request_id=request_id,
                        parameters={'Streaming': FLAGS.streaming})

                #Wait for server to close the stream
                triton_client.stop_stream()

                # Parse the responses
                while True:
                    try:
                        result = user_data._completed_requests.get(block=False)
                    except Exception:
                        break

                    if type(result) == InferenceServerException:
                        print("Received an error from server:")
                        print(result)
                    else:
                        output_ids = result.as_numpy('output_ids')
                        sequence_lengths = result.as_numpy('sequence_length')
                        if output_ids is not None:
                            # Only one beam is supported
                            tokens = list(output_ids[0][0])
                            actual_output_ids[
                                0] = actual_output_ids[0] + tokens
                        else:
                            print("Got cancellation response from server")
            else:
                # Send request
                triton_client.async_infer(
                    'tensorrt_llm',
                    inputs,
                    request_id=request_id,
                    callback=partial(callback, user_data),
                    parameters={'Streaming': FLAGS.streaming})

                if stop_inputs is not None:

                    time.sleep(FLAGS.stop_after_ms / 1000.0)

                    triton_client.async_infer(
                        'tensorrt_llm',
                        stop_inputs,
                        request_id=request_id,
                        callback=partial(callback, user_data),
                        parameters={'Streaming': FLAGS.streaming})

                processed_count = 0
                expected_responses = 1 + (1 if stop_inputs is not None else 0)
                while processed_count < expected_responses:
                    try:
                        result = user_data._completed_requests.get()
                        print("Got completed request", flush=True)
                    except Exception:
                        break

                    if type(result) == InferenceServerException:
                        print("Received an error from server:")
                        print(result)
                    else:
                        output_ids = result.as_numpy('output_ids')
                        sequence_lengths = result.as_numpy('sequence_length')
                        if output_ids is not None:
                            for beam_output_ids in output_ids[0]:
                                tokens = list(beam_output_ids)
                                actual_output_ids.append(tokens)
                        else:
                            print("Got cancellation response from server")

                    processed_count = processed_count + 1
        except Exception as e:
            print("channel creation failed: " + str(e))
            sys.exit()

        passed = True

        for beam in range(FLAGS.beam_width):
            seq_len = sequence_lengths[0][
                beam] if not FLAGS.streaming else len(actual_output_ids[beam])
            # These should be equal when input IDs are excluded from output
            output_ids_w_prompt = actual_output_ids[beam][:seq_len]
            output_ids_wo_prompt = (
                output_ids_w_prompt if FLAGS.exclude_input_in_output else
                output_ids_w_prompt[input_ids_data.shape[1]:])
            if tokenizer != None:
                output_text = tokenizer.decode(output_ids_wo_prompt)
                print(f'Input: {FLAGS.text}')
                print(f'Output beam {beam}: {output_text}')

            print("output_ids = ", output_ids_w_prompt)
            if (FLAGS.check_output and beam == 0):
                passed = (output_ids_w_prompt == expected_output_ids)
                print("expected_output_ids = ", expected_output_ids)
                print("\n=====")
                print("PASS!" if passed else "FAIL!")
                print("=====")

        sys.exit(not passed)
