#!/usr/bin/python

import sys

sys.path.append('../')

import argparse
import json
import sys
from datetime import datetime

import numpy as np
import tritonclient.http as httpclient
from utils import token_encoder, utils

MERGES_FILE = "gpt2-merges.txt"
VOCAB_FILE = "gpt2-vocab.json"


def test_performance(client, input_start_ids, input_lens):
    model_name = "tensorrt_llm"

    print(f"[INFO] Warm up for benchmarking.")
    for i in range(10):
        output0_len = np.ones_like([[1]]).astype(np.int32) * 100
        inputs = [
            utils.prepare_tensor("input_ids", input_start_ids[0],
                                 FLAGS.protocol),
            utils.prepare_tensor("input_lengths", input_lens[i],
                                 FLAGS.protocol),
            utils.prepare_tensor("request_output_len", output0_len,
                                 FLAGS.protocol),
        ]
        client.infer(model_name, inputs, request_id=str(i))

    print(f"[INFO] Start benchmarking on {len(input_start_ids)} prompts.")
    latency = 0
    async_requests = []
    output_len = [5, 60, 100]
    start_time = datetime.now()
    for i, ids in enumerate(input_start_ids):
        output0_len = np.ones_like([[1]]).astype(np.int32) * output_len[i % 3]
        inputs = [
            utils.prepare_tensor("input_ids", ids, FLAGS.protocol),
            utils.prepare_tensor("input_lengths", input_lens[i],
                                 FLAGS.protocol),
            utils.prepare_tensor("request_output_len", output0_len,
                                 FLAGS.protocol),
        ]

        async_requests.append(
            client.async_infer(model_name, inputs, request_id=str(i)))

    utils.get_http_results(async_requests)

    stop_time = datetime.now()
    latency = (stop_time - start_time).total_seconds() * 1000.0
    latency = round(latency, 3)
    print(f"[INFO] Total Latency: {latency} ms")


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
        choices=['http', 'grpc'],
        help='Protocol ("http"/"grpc") used to ' +
        'communicate with inference service. Default is "http".')
    parser.add_argument('-c',
                        '--concurrency',
                        type=int,
                        default=128,
                        required=False,
                        help='Specify concurrency')

    parser.add_argument('-o',
                        '--output_len',
                        type=int,
                        default=100,
                        required=False,
                        help='Specify output length')
    parser.add_argument('--dataset',
                        type=str,
                        required=True,
                        help='Dataset path used for the test.')

    FLAGS = parser.parse_args()
    if FLAGS.url is None:
        FLAGS.url = "localhost:8000" if FLAGS.protocol == "http" else "localhost:8001"

    # For the HTTP client, need to specify large enough concurrency to
    # issue all the inference requests to the server in parallel. For
    # this example we want to be able to send 2 requests concurrently.
    try:
        client = httpclient.InferenceServerClient(
            url=FLAGS.url, concurrency=FLAGS.concurrency)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit(1)

    encoder = token_encoder.get_encoder(VOCAB_FILE, MERGES_FILE)
    input_start_ids = []
    input_lens = []
    with open(FLAGS.dataset) as f:
        for line in f:
            line = json.loads(line)
            line = encoder.encode(line['prompt'])
            input_start_ids.append(np.array([line], np.int32))
            input_lens.append(np.array([[len(line)]], np.int32))

    test_performance(client, input_start_ids, input_lens)
