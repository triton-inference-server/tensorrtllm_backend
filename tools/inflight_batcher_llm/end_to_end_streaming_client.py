#!/usr/bin/python

import os
import sys
from functools import partial

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import argparse
import queue
import sys

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException
from utils import utils


class UserData:

    def __init__(self):
        self._completed_requests = queue.Queue()


def callback(user_data, result, error):
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)
        output = result.as_numpy('text_output')
        print(output[0], flush=True)


def test(triton_client, prompt):
    model_name = "ensemble"

    input0 = [[prompt]]
    input0_data = np.array(input0).astype(object)
    output0_len = np.ones_like(input0).astype(np.uint32) * FLAGS.output_len
    bad_words_list = np.array([[""]], dtype=object)
    stop_words_list = np.array([[""]], dtype=object)
    streaming = [[FLAGS.streaming]]
    streaming_data = np.array(streaming, dtype=bool)

    inputs = [
        utils.prepare_tensor("text_input", input0_data, FLAGS.protocol),
        utils.prepare_tensor("max_tokens", output0_len, FLAGS.protocol),
        utils.prepare_tensor("bad_words", bad_words_list, FLAGS.protocol),
        utils.prepare_tensor("stop_words", stop_words_list, FLAGS.protocol),
        utils.prepare_tensor("stream", streaming_data, FLAGS.protocol),
    ]

    user_data = UserData()
    # Establish stream
    triton_client.start_stream(callback=partial(callback, user_data))
    # Send request
    triton_client.async_stream_infer(model_name, inputs)

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
            result.as_numpy('text_output')


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

    parser.add_argument('-p',
                        '--prompt',
                        type=str,
                        required=True,
                        help='Input prompt.')
    parser.add_argument(
        "-S",
        "--streaming",
        action="store_true",
        required=False,
        default=False,
        help="Enable streaming mode. Default is False.",
    )

    parser.add_argument(
        '-i',
        '--protocol',
        type=str,
        required=False,
        default='grpc',
        choices=['grpc'],
        help='Protocol ("http"/"grpc") used to ' +
        'communicate with inference service. Default is "http".')

    parser.add_argument('-o',
                        '--output_len',
                        type=int,
                        default=100,
                        required=False,
                        help='Specify output length')

    FLAGS = parser.parse_args()
    if FLAGS.url is None:
        FLAGS.url = "localhost:8000" if FLAGS.protocol == "http" else "localhost:8001"

    try:
        client = grpcclient.InferenceServerClient(url=FLAGS.url)
    except Exception as e:
        print("client creation failed: " + str(e))
        sys.exit(1)

    test(client, FLAGS.prompt)
