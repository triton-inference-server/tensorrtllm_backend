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


def test_performance(client, input_start_ids, input_lens, output_lens, delays,
                     FLAGS):
    model_name = "tensorrt_llm"

    print(f"[INFO] Warm up for benchmarking.")
    for i in range(10):
        output0_len = np.ones_like([[1]]).astype(np.int32) * 100
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
        output0_len = np.ones_like([[1]]).astype(np.int32) * output_lens[i]
        end_id = np.ones_like([[1]]).astype(np.int32) * -1
        inputs = [
            utils.prepare_tensor("input_ids", ids, FLAGS.protocol),
            utils.prepare_tensor("input_lengths", input_lens[i],
                                 FLAGS.protocol),
            utils.prepare_tensor("request_output_len", output0_len,
                                 FLAGS.protocol),
            utils.prepare_tensor("end_id", end_id, FLAGS.protocol),
        ]

        time.sleep(delays[i])

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
    subparsers = parser.add_subparsers(dest='workload')

    parser_dataset = subparsers.add_parser('dataset')
    parser_dataset.add_argument('--dataset',
                                type=str,
                                required=True,
                                help='Dataset path used for the test.')
    parser_dataset.add_argument('--tokenizer-dir',
                                type=str,
                                required=True,
                                help='Specify tokenizer directory')
    parser_dataset.add_argument('--tokenizer-type',
                                type=str,
                                default='auto',
                                required=False,
                                choices=['auto', 't5', 'llama'],
                                help='Specify tokenizer type')
    parser_dataset.add_argument(
        '--op-tokens-per-word',
        type=float,
        default=1.3,
        required=False,
        help=
        'Specify op tokens/word ratio. Useful to have model generate exactly as many tokens as needed by the dataset'
    )

    parser_token_norm_dist = subparsers.add_parser('token-norm-dist')
    parser_token_norm_dist.add_argument(
        '--input-mean',
        type=int,
        required=True,
        help='normal dist mean for input tokens')
    parser_token_norm_dist.add_argument(
        '--input-stdev',
        type=int,
        required=True,
        help='normal dist stdev for input tokens')
    parser_token_norm_dist.add_argument(
        '--output-mean',
        type=int,
        required=True,
        help='normal dist mean for output tokens')
    parser_token_norm_dist.add_argument(
        '--output-stdev',
        type=int,
        required=True,
        help='normal dist stdev for output tokens')

    parser_token_from_hist = subparsers.add_parser('token-from-histogram')
    parser_token_from_hist.add_argument(
        '--histogram-key',
        type=str,
        required=True,
        help='key to retrieve histogram buckets,freqs defined in utils')

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
    parser.add_argument('--max-input-len',
                        type=int,
                        required=True,
                        help='Specify max input length')
    parser.add_argument('--request-rate',
                        type=float,
                        required=False,
                        help="# of reqs/sec. -1 indicates SOL/Offline",
                        default=-1.0)
    parser.add_argument('--time-delay-dist',
                        type=str,
                        required=False,
                        choices=["constant", "exponential_dist"],
                        default="exponential_dist",
                        help="# of reqs/sec. -1 indicates SOL/Offline")
    parser.add_argument(
        '--dump-perfetto-trace',
        action="store_true",
        required=False,
        default=False,
        help=
        'Dumps trace of requests in a json (perfetto.json) to be visualized in perfetto'
    ),
    parser.add_argument('--op-stats-csv',
                        type=str,
                        default=None,
                        help='csv filename to dump stats'),
    parser.add_argument(
        "--exclude-input-in-output",
        action="store_true",
        required=False,
        default=False,
        help="Expect that output IDs do not contain input IDs",
    )
    parser.add_argument(
        '--num-requests',
        type=int,
        required=False,
        default=30000,
        help=
        'For dataset, requests = min(dataset, num_requests). number of requests to be generated by the client'
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

    if FLAGS.request_rate == -1:
        mean_time_bet_reqs = 0
    else:
        mean_time_bet_reqs = 1.0 / FLAGS.request_rate

    input_start_ids = []
    input_lens = []
    output_lens = []
    ratio = []

    print(FLAGS.workload)
    if FLAGS.workload == "dataset":

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
        prompt_cnt = 0

        with open(FLAGS.dataset, 'r') as f:
            data_dict = json.load(f)
            for req in data_dict:
                prompt = req['input'] + ' ' + req['instruction']
                output = req['output']
                line = tokenizer.encode(prompt)
                if len(line) > FLAGS.max_input_len:
                    continue

                prompt_cnt += 1
                if prompt_cnt > FLAGS.num_requests:
                    break

                input_start_ids.append(np.array([line], np.int32))
                input_lens.append(np.array([[len(line)]], np.int32))
                output_lens.append(
                    int(len(output.split(' ')) * FLAGS.op_tokens_per_word))
                prompt_tokens = len(line)
                prompt_words = len(prompt.split())
                ratio.append(prompt_tokens / prompt_words)

        print("Tokenizer: Tokens per word = ", round(np.mean(ratio), 3))
        num_reqs = len(input_lens)
        delays = utils.get_list_of_delays(FLAGS.time_delay_dist,
                                          mean_time_bet_reqs, num_reqs)
        test_performance(client, input_start_ids, input_lens, output_lens,
                         delays, FLAGS)

    elif FLAGS.workload == "token-norm-dist":
        input_lens = utils.get_norm_dist_tokens(FLAGS.input_mean,
                                                FLAGS.input_stdev,
                                                FLAGS.num_requests)
        pruned_ip_list = [
            ip_len for ip_len in input_lens if ip_len <= FLAGS.max_input_len
        ]
        num_reqs = len(pruned_ip_list)
        ip_lens_2d_array = [
            np.array([[ip_len]], np.int32) for ip_len in pruned_ip_list
        ]
        output_lens = utils.get_norm_dist_tokens(FLAGS.output_mean,
                                                 FLAGS.output_stdev, num_reqs)
        delays = utils.get_list_of_delays(FLAGS.time_delay_dist,
                                          mean_time_bet_reqs, num_reqs)

        input_start_ids = utils.gen_random_start_ids(pruned_ip_list)
        test_performance(client, input_start_ids, ip_lens_2d_array,
                         output_lens, delays, FLAGS)

    elif FLAGS.workload == "token-from-histogram":
        input_lens_orig = utils.get_token_list_from_histogram(
            FLAGS.histogram_key + "_ip")
        output_lens_orig = utils.get_token_list_from_histogram(
            FLAGS.histogram_key + "_op")

        final_lens = min(len(input_lens_orig), len(output_lens_orig))
        input_lens = input_lens_orig[:final_lens]
        output_lens = output_lens_orig[:final_lens]

        num_reqs = len(input_lens)
        ip_lens_2d_array = [
            np.array([[ip_len]], np.int32) for ip_len in input_lens
        ]
        output_lens = utils.get_token_list_from_histogram(FLAGS.histogram_key +
                                                          "_op")
        print(len(input_lens), len(output_lens))
        assert (len(input_lens) == len(output_lens))

        delays = utils.get_list_of_delays(FLAGS.time_delay_dist,
                                          mean_time_bet_reqs, num_reqs)

        input_start_ids = utils.gen_random_start_ids(input_lens)
        test_performance(client, input_start_ids, ip_lens_2d_array,
                         output_lens, delays, FLAGS)
