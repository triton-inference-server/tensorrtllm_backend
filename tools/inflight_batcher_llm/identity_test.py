#!/usr/bin/python

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import argparse
import json
import sys
import time
from datetime import datetime
from functools import partial

import numpy as np
from transformers import AutoTokenizer, LlamaTokenizer, T5Tokenizer
from utils import utils


def callback(user_data, start_time, req_id, result, error):
    user_data._completed_requests.put((result, error))
    stop_time = datetime.now()
    latency = (stop_time - start_time).total_seconds() * 1000.0
    latency = round(latency, 3)
    user_data._latencies.append(latency)
    user_data._latency_dict[req_id] = latency
    user_data._start_time_dict[req_id] = start_time
    user_data._stop_time_dict[req_id] = stop_time


def test_performance(client, input_start_ids, input_lens, output_lens, FLAGS):
    model_name = "tensorrt_llm"

    print(f"[INFO] Warm up for benchmarking.")
    for i in range(10):
        output0_len = np.ones_like([[1]]).astype(np.uint32) * 100
        inputs = [
            utils.prepare_tensor("input_ids", input_start_ids[0],
                                 FLAGS.protocol),
            utils.prepare_tensor("input_lengths", input_lens[0],
                                 FLAGS.protocol),
            utils.prepare_tensor("request_output_len", output0_len,
                                 FLAGS.protocol),
        ]
        client.infer(model_name, inputs, request_id=str(i))

    print(f"[INFO] Start benchmarking on {len(input_start_ids)} prompts.")
    latency = 0
    async_requests = []
    start_time = datetime.now()
    user_data = utils.UserData()
    for i, ids in enumerate(input_start_ids):
        output0_len = np.ones_like([[1]]).astype(np.uint32) * output_lens[i]
        inputs = [
            utils.prepare_tensor("input_ids", ids, FLAGS.protocol),
            utils.prepare_tensor("input_lengths", input_lens[i],
                                 FLAGS.protocol),
            utils.prepare_tensor("request_output_len", output0_len,
                                 FLAGS.protocol),
        ]

        if len(FLAGS.time_bet_reqs) == 1:
            time.sleep(FLAGS.time_bet_reqs[0])
        else:
            time.sleep(FLAGS.time_bet_reqs[i % len(FLAGS.time_bet_reqs)])

        if FLAGS.protocol == "http":
            async_requests.append(
                client.async_infer(model_name, inputs, request_id=str(i)))
        elif FLAGS.protocol == "grpc":
            async_requests.append(
                client.async_infer(model_name,
                                   inputs,
                                   callback=partial(callback, user_data,
                                                    datetime.now(), i),
                                   request_id=str(i)))

    try:
        if FLAGS.protocol == "http":
            utils.get_http_results(async_requests)
        elif FLAGS.protocol == "grpc":
            responses = utils.get_grpc_results(user_data, len(input_start_ids))
        else:
            raise RuntimeError("Invalid protocol")

        stop_time = datetime.now()
        latency = (stop_time - start_time).total_seconds() * 1000.0
        latency = round(latency, 3)
        print(f"[INFO] Total Latency: {latency} ms")

        # TODO(kaiyu): support `extract_print_stats` for http
        if FLAGS.protocol == "grpc":
            request_latencies = 0.0
            for latency in user_data._latencies:
                request_latencies += latency
            print(f"[INFO] Total request latencies: {request_latencies} ms")

            ip_token_len_list = []
            for ip in input_lens:
                ip_token_len_list.append(
                    ip[0][0])  #for some reason, two level nesting

            utils.extract_print_stats(ip_token_len_list, responses, user_data,
                                      FLAGS)

    except Exception as e:
        print("Failed receiving responses: " + str(e))
        sys.exit(1)


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
    parser.add_argument('--max_input_len',
                        type=int,
                        required=True,
                        help='Specify max input length')

    parser.add_argument('--dataset',
                        type=str,
                        required=True,
                        help='Dataset path used for the test.')
    parser.add_argument('--tokenizer_dir',
                        type=str,
                        required=True,
                        help='Specify tokenizer directory')
    parser.add_argument('--tokenizer_type',
                        type=str,
                        default='auto',
                        required=False,
                        choices=['auto', 't5', 'llama'],
                        help='Specify tokenizer type')
    parser.add_argument(
        '--time_bet_reqs',
        type=float,
        required=False,
        nargs='+',
        help="Input time(s) in (secs) bet requests separated by spaces",
        default=[0])
    parser.add_argument(
        '--dump_perfetto_trace',
        action="store_true",
        required=False,
        default=False,
        help=
        'Dumps trace of requests in a json (perfetto.json) to be visualized in perfetto'
    ),
    parser.add_argument('--op_stats_csv',
                        type=str,
                        default=None,
                        help='csv filename to dump stats'),
    parser.add_argument(
        '--op_tokens_per_word',
        type=float,
        default=1.3,
        required=False,
        help=
        'Specify op tokens/word ratio. Useful to have model generate as many number of words as in dataset'
    )
    parser.add_argument(
        "--exclude_input_in_output",
        action="store_true",
        required=False,
        default=False,
        help="Expect that output IDs do not contain input IDs",
    )

    FLAGS = parser.parse_args()
    if FLAGS.url is None:
        FLAGS.url = "localhost:8000" if FLAGS.protocol == "http" else "localhost:8001"

    try:
        client = utils.create_inference_server_client(
            FLAGS.protocol,
            FLAGS.url,
            concurrency=FLAGS.concurrency,
            verbose=FLAGS.verbose)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit(1)

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

    input_start_ids = []
    input_lens = []
    output_lens = []
    ratio = []
    with open(FLAGS.dataset, 'r') as f:
        data_dict = json.load(f)
        for req in data_dict:
            prompt = req['input'] + ' ' + req['instruction']
            output = req['output']
            line = tokenizer.encode(prompt)
            if len(line) > FLAGS.max_input_len:
                continue
            input_start_ids.append(np.array([line], np.int32))
            input_lens.append(np.array([[len(line)]], np.int32))
            output_lens.append(
                int(len(output.split(' ')) * FLAGS.op_tokens_per_word))
            prompt_tokens = len(line)
            prompt_words = len(prompt.split())
            ratio.append(prompt_tokens / prompt_words)

    print("Tokens per word: ", round(np.mean(ratio), 3))
    test_performance(client, input_start_ids, input_lens, output_lens, FLAGS)
