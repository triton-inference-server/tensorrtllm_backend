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
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException, np_to_triton_dtype


def prepare_tensor(name, input, protocol):
    client_util = httpclient if protocol == "http" else grpcclient
    t = client_util.InferInput(name, input.shape,
                               np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t


class UserData:

    def __init__(self):
        self._completed_requests = queue.Queue()


def callback(user_data, result, error):
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)
        output = result.as_numpy('text_output')
        print(output, flush=True)


def test(triton_client, prompt, request_id, repetition_penalty,
         presence_penalty, temperatuure, stop_words, bad_words,
         embedding_bias_words, embedding_bias_weights):
    model_name = "ensemble"

    input0 = [[prompt]]
    input0_data = np.array(input0).astype(object)
    output0_len = np.ones_like(input0).astype(np.uint32) * FLAGS.output_len
    streaming = [[FLAGS.streaming]]
    streaming_data = np.array(streaming, dtype=bool)
    beam_width = [[FLAGS.beam_width]]
    beam_width_data = np.array(beam_width, dtype=np.uint32)
    temperature = [[FLAGS.temperature]]
    temperature_data = np.array(temperature, dtype=np.float32)

    inputs = [
        prepare_tensor("text_input", input0_data, FLAGS.protocol),
        prepare_tensor("max_tokens", output0_len, FLAGS.protocol),
        prepare_tensor("stream", streaming_data, FLAGS.protocol),
        prepare_tensor("beam_width", beam_width_data, FLAGS.protocol),
        prepare_tensor("temperature", temperature_data, FLAGS.protocol),
    ]

    if bad_words:
        bad_words_list = np.array([bad_words], dtype=object)
        inputs += [prepare_tensor("bad_words", bad_words_list, FLAGS.protocol)]

    if stop_words:
        stop_words_list = np.array([stop_words], dtype=object)
        inputs += [
            prepare_tensor("stop_words", stop_words_list, FLAGS.protocol)
        ]

    if repetition_penalty is not None:
        repetition_penalty = [[repetition_penalty]]
        repetition_penalty_data = np.array(repetition_penalty,
                                           dtype=np.float32)
        inputs += [
            prepare_tensor("repetition_penalty", repetition_penalty_data,
                           FLAGS.protocol),
        ]

    if presence_penalty is not None:
        presence_penalty = [[presence_penalty]]
        presence_penalty_data = np.array(presence_penalty, dtype=np.float32)
        inputs += [
            prepare_tensor("presence_penalty", presence_penalty_data,
                           FLAGS.protocol),
        ]

    if (embedding_bias_words is not None and embedding_bias_weights is None
        ) or (embedding_bias_words is None
              and embedding_bias_weights is not None):
        assert 0, "Both embedding bias words and weights must be specified"

    if (embedding_bias_words is not None
            and embedding_bias_weights is not None):
        assert len(embedding_bias_words) == len(
            embedding_bias_weights
        ), "Embedding bias weights and words must have same length"
        embedding_bias_words_data = np.array([embedding_bias_words],
                                             dtype=object)
        embedding_bias_weights_data = np.array([embedding_bias_weights],
                                               dtype=np.float32)
        inputs.append(
            prepare_tensor("embedding_bias_words", embedding_bias_words_data,
                           FLAGS.protocol))
        inputs.append(
            prepare_tensor("embedding_bias_weights",
                           embedding_bias_weights_data, FLAGS.protocol), )

    user_data = UserData()
    # Establish stream
    triton_client.start_stream(callback=partial(callback, user_data))
    # Send request
    triton_client.async_stream_infer(model_name, inputs, request_id=request_id)

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
        "--repetition-penalty",
        type=float,
        required=False,
        default=None,
        help="The repetition penalty value",
    )

    parser.add_argument(
        "--presence-penalty",
        type=float,
        required=False,
        default=None,
        help="The presence penalty value",
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
                        '--output-len',
                        type=int,
                        default=100,
                        required=False,
                        help='Specify output length')

    parser.add_argument('--request-id',
                        type=str,
                        default='1',
                        required=False,
                        help='The request_id for the stop request')

    parser.add_argument('--stop-words',
                        nargs='+',
                        default=[],
                        help='The stop words')

    parser.add_argument('--bad-words',
                        nargs='+',
                        default=[],
                        help='The bad words')

    parser.add_argument('--embedding-bias-words',
                        nargs='+',
                        default=[],
                        help='The biased words')

    parser.add_argument('--embedding-bias-weights',
                        nargs='+',
                        default=[],
                        help='The biased words weights')

    FLAGS = parser.parse_args()
    if FLAGS.url is None:
        FLAGS.url = "localhost:8000" if FLAGS.protocol == "http" else "localhost:8001"

    embedding_bias_words = FLAGS.embedding_bias_words if FLAGS.embedding_bias_words else None
    embedding_bias_weights = FLAGS.embedding_bias_weights if FLAGS.embedding_bias_weights else None

    try:
        client = grpcclient.InferenceServerClient(url=FLAGS.url)
    except Exception as e:
        print("client creation failed: " + str(e))
        sys.exit(1)

    test(client, FLAGS.prompt, FLAGS.request_id, FLAGS.repetition_penalty,
         FLAGS.presence_penalty, FLAGS.temperature, FLAGS.stop_words,
         FLAGS.bad_words, embedding_bias_words, embedding_bias_weights)
