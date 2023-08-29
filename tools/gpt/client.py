#!/usr/bin/python

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import argparse
from datetime import datetime

import numpy as np
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

    if FLAGS.url is None:
        FLAGS.url = "localhost:8000" if FLAGS.protocol == "http" else "localhost:8001"

    encoder = token_encoder.get_encoder(VOCAB_FILE, MERGES_FILE)
    line = encoder.encode(FLAGS.text)
    input_start_ids = np.array([line], np.int32)
    input_len = np.array([[len(line)]], np.int32)
    inputs = utils.prepare_inputs(input_start_ids, input_len, FLAGS)

    start_time = datetime.now()

    with utils.create_inference_server_client(FLAGS.protocol,
                                              FLAGS.url,
                                              concurrency=FLAGS.concurrency,
                                              verbose=FLAGS.verbose) as client:
        results = utils.send_requests('tensorrt_llm',
                                      inputs,
                                      client,
                                      request_parallelism=1)
    output_ids = results[0].as_numpy("output_ids")

    stop_time = datetime.now()
    latencies = (stop_time - start_time).total_seconds() * 1000.0
    print(f"[INFO] Latency: {latencies} ms")

    output_ids = output_ids.reshape(
        (output_ids.size, )).tolist()[input_start_ids.shape[1]:]
    output_text = encoder.decode(output_ids)
    print(f'Input: {FLAGS.text}')
    print(f'Output: {output_text}')
