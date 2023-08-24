#!/usr/bin/python

import sys

sys.path.append('../')

import argparse
import json
import sys
from datetime import datetime

import numpy as np
import tritonclient.http as httpclient
from utils import utils


def test_functionality(client, prompts):
    print(f"[INFO] Start testing on {len(prompts)} prompts.")
    for i, prompt in enumerate(prompts):

        # 1. Ensemble models manually: preprocessing -> tensorrt_llm -> postprocessing
        model_name = 'preprocessing'
        input0 = [[prompt]]
        input0_data = np.array(input0).astype(object)
        output0_len = np.ones_like(input0).astype(np.uint32) * FLAGS.output_len

        inputs = [
            utils.prepare_tensor("QUERY", input0_data, FLAGS.protocol),
            utils.prepare_tensor("REQUEST_OUTPUT_LEN", output0_len,
                                 FLAGS.protocol),
        ]
        result = client.infer(model_name, inputs, request_id=str(i))
        output0 = result.as_numpy("INPUT_ID")
        output1 = result.as_numpy("REQUEST_INPUT_LEN")
        output2 = result.as_numpy("REQUEST_OUTPUT_LEN")

        model_name = "tensorrt_llm"
        inputs = [
            utils.prepare_tensor("input_ids", output0, FLAGS.protocol),
            utils.prepare_tensor("input_lengths", output1, FLAGS.protocol),
            utils.prepare_tensor("request_output_len", output2,
                                 FLAGS.protocol),
        ]
        result = client.infer(model_name, inputs, request_id=str(i))
        output0 = result.as_numpy("output_ids")

        model_name = "postprocessing"
        inputs = [
            utils.prepare_tensor("TOKENS_BATCH", output0, FLAGS.protocol)
        ]
        inputs[0].set_data_from_numpy(output0)

        result = client.infer(model_name, inputs, request_id=str(i))
        output0 = result.as_numpy("OUTPUT")

        # 2. Use ensemble model
        model_name = "ensemble"
        input0 = [[prompt]]
        input0_data = np.array(input0).astype(object)
        output0_len = np.ones_like(input0).astype(np.uint32) * FLAGS.output_len

        inputs = [
            utils.prepare_tensor("INPUT_0", input0_data, FLAGS.protocol),
            utils.prepare_tensor("INPUT_1", output0_len, FLAGS.protocol),
        ]

        result = client.infer(model_name, inputs, request_id=str(i))

        # 3. Check the results between manually ensembled models and the ensemble model
        ensemble_output = result.as_numpy('OUTPUT_0')
        assert output0 == ensemble_output
        print('Response: {}'.format(result.get_response()))
        print('Output: {}'.format(ensemble_output))
    print(f"[INFO] Functionality test succeed.")


def test_performance(client, prompts):
    model_name = "ensemble"

    print(f"[INFO] Warm up for benchmarking.")
    for i in range(10):
        input0 = [[prompts[0]]]
        input0_data = np.array(input0).astype(object)
        output0_len = np.ones_like(input0).astype(np.uint32) * FLAGS.output_len

        inputs = [
            utils.prepare_tensor("INPUT_0", input0_data, FLAGS.protocol),
            utils.prepare_tensor("INPUT_1", output0_len, FLAGS.protocol),
        ]

        client.infer(model_name, inputs, request_id=str(i))

    #sys.exit(1)
    print(f"[INFO] Start benchmarking on {len(prompts)} prompts.")
    latency = 0
    async_requests = []
    output_len = [5, 60, 100]
    start_time = datetime.now()
    for i, prompt in enumerate(prompts):
        input0 = [[prompt]]
        input0_data = np.array(input0).astype(object)
        output0_len = np.ones_like(input0).astype(
            np.uint32) * output_len[i % 3]

        inputs = [
            utils.prepare_tensor("INPUT_0", input0_data, FLAGS.protocol),
            utils.prepare_tensor("INPUT_1", output0_len, FLAGS.protocol),
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

    prompts = []
    with open(FLAGS.dataset) as f:
        for line in f:
            line = json.loads(line)
            prompts.append(line['prompt'])

    test_functionality(client, prompts)
    test_performance(client, prompts)
