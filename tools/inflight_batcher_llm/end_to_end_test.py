#!/usr/bin/python

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import argparse
import json
import sys
from datetime import datetime
from functools import partial

import numpy as np
from utils import utils


def callback(user_data, start_time, result, error):
    user_data._completed_requests.put((result, error))
    stop_time = datetime.now()
    latency = (stop_time - start_time).total_seconds() * 1000.0
    latency = round(latency, 3)
    user_data._latencies.append(latency)


def test_functionality(client, prompts, output_lens):
    print(f"[INFO] Start testing on {len(prompts)} prompts.")
    for i, prompt in enumerate(prompts):

        # 1. Ensemble models manually: preprocessing -> tensorrt_llm -> postprocessing
        model_name = 'preprocessing'
        input0 = [[prompt]]
        input0_data = np.array(input0).astype(object)
        output0_len = np.ones_like(input0).astype(np.int32) * output_lens[i]
        bad_words_list = np.array([[""]], dtype=object)
        stop_words_list = np.array([[""]], dtype=object)

        inputs = [
            utils.prepare_tensor("QUERY", input0_data, FLAGS.protocol),
            utils.prepare_tensor("BAD_WORDS_DICT", bad_words_list,
                                 FLAGS.protocol),
            utils.prepare_tensor("STOP_WORDS_DICT", stop_words_list,
                                 FLAGS.protocol),
            utils.prepare_tensor("REQUEST_OUTPUT_LEN", output0_len,
                                 FLAGS.protocol),
        ]
        result = client.infer(model_name, inputs, request_id=str(i))
        output0 = result.as_numpy("INPUT_ID")
        output1 = result.as_numpy("REQUEST_INPUT_LEN")
        output2 = result.as_numpy("REQUEST_OUTPUT_LEN")
        output_end_id = result.as_numpy("OUT_END_ID")
        output_pad_id = result.as_numpy("OUT_PAD_ID")

        model_name = "tensorrt_llm"
        inputs = [
            utils.prepare_tensor("input_ids", output0, FLAGS.protocol),
            utils.prepare_tensor("input_lengths", output1, FLAGS.protocol),
            utils.prepare_tensor("request_output_len", output2,
                                 FLAGS.protocol),
            utils.prepare_tensor("end_id", output_end_id, FLAGS.protocol),
            utils.prepare_tensor("pad_id", output_pad_id, FLAGS.protocol),
        ]
        result = client.infer(model_name, inputs, request_id=str(i))
        output0 = result.as_numpy("output_ids").astype(np.int32)
        seq_lengths = result.as_numpy("sequence_length")
        cum_log_probs = result.as_numpy("cum_log_probs").astype(np.float32)
        output_log_probs = result.as_numpy("output_log_probs").astype(
            np.float32)
        context_logits = result.as_numpy("context_logits").astype(np.float32)
        generation_logits = result.as_numpy("generation_logits").astype(
            np.float32)

        print(f"context_logits.shape: {context_logits.shape}")
        print(f"generation_logits.shape: {generation_logits.shape}")

        model_name = "postprocessing"
        inputs = [
            utils.prepare_tensor("TOKENS_BATCH", output0, FLAGS.protocol),
            utils.prepare_tensor("SEQUENCE_LENGTH", seq_lengths,
                                 FLAGS.protocol),
            utils.prepare_tensor("CUM_LOG_PROBS", cum_log_probs,
                                 FLAGS.protocol),
            utils.prepare_tensor("OUTPUT_LOG_PROBS", output_log_probs,
                                 FLAGS.protocol),
            utils.prepare_tensor("CONTEXT_LOGITS", context_logits,
                                 FLAGS.protocol),
            utils.prepare_tensor("GENERATION_LOGITS", generation_logits,
                                 FLAGS.protocol)
        ]
        inputs[0].set_data_from_numpy(output0)
        inputs[1].set_data_from_numpy(seq_lengths)
        inputs[2].set_data_from_numpy(cum_log_probs)
        inputs[3].set_data_from_numpy(output_log_probs)
        inputs[4].set_data_from_numpy(context_logits)
        inputs[5].set_data_from_numpy(generation_logits)

        result = client.infer(model_name, inputs, request_id=str(i))
        output0 = result.as_numpy("OUTPUT")

        # 2. Use ensemble model
        model_name = "ensemble"
        input0 = [[prompt]]
        input0_data = np.array(input0).astype(object)
        output0_len = np.ones_like(input0).astype(np.int32) * output_lens[i]
        bad_words_list = np.array([[""]], dtype=object)
        stop_words_list = np.array([[""]], dtype=object)

        inputs = [
            utils.prepare_tensor("text_input", input0_data, FLAGS.protocol),
            utils.prepare_tensor("max_tokens", output0_len, FLAGS.protocol),
            utils.prepare_tensor("bad_words", bad_words_list, FLAGS.protocol),
            utils.prepare_tensor("stop_words", stop_words_list,
                                 FLAGS.protocol),
        ]

        result = client.infer(model_name, inputs, request_id=str(i))

        # 3. Check the results between manually ensembled models and the ensemble model
        ensemble_output = result.as_numpy('text_output')
        ensemble_cum_log_probs = result.as_numpy('cum_log_probs')
        ensemble_output_log_probs = result.as_numpy('output_log_probs')
        ensemble_context_logits = result.as_numpy('context_logits')
        ensemble_generation_logits = result.as_numpy('generation_logits')

        assert output0 == ensemble_output
        assert cum_log_probs == ensemble_cum_log_probs
        assert (output_log_probs == ensemble_output_log_probs).all()
        assert (context_logits == ensemble_context_logits).all()
        assert (generation_logits == ensemble_generation_logits).all()

        if FLAGS.verbose:
            print('Response: {}'.format(result.get_response()))
            print('Output: {}'.format(ensemble_output))
    print(f"[INFO] Functionality test succeed.")


def test_performance(client, prompts, output_lens):
    model_name = "ensemble"

    print(f"[INFO] Warm up for benchmarking.")
    for i in range(min(10, len(prompts))):
        input0 = [[prompts[0]]]
        input0_data = np.array(input0).astype(object)
        output0_len = np.ones_like(input0).astype(np.int32) * output_lens[i]
        bad_words_list = np.array([[""]], dtype=object)
        stop_words_list = np.array([[""]], dtype=object)

        inputs = [
            utils.prepare_tensor("text_input", input0_data, FLAGS.protocol),
            utils.prepare_tensor("max_tokens", output0_len, FLAGS.protocol),
            utils.prepare_tensor("bad_words", bad_words_list, FLAGS.protocol),
            utils.prepare_tensor("stop_words", stop_words_list,
                                 FLAGS.protocol),
        ]

        client.infer(model_name, inputs, request_id=str(i))

    print(f"[INFO] Start benchmarking on {len(prompts)} prompts.")
    latency = 0
    async_requests = []
    start_time = datetime.now()
    user_data = utils.UserData()
    for i, prompt in enumerate(prompts):
        input0 = [[prompt]]
        input0_data = np.array(input0).astype(object)
        output0_len = np.ones_like(input0).astype(np.int32) * output_lens[i]
        bad_words_list = np.array([[""]], dtype=object)
        stop_words_list = np.array([[""]], dtype=object)

        inputs = [
            utils.prepare_tensor("text_input", input0_data, FLAGS.protocol),
            utils.prepare_tensor("max_tokens", output0_len, FLAGS.protocol),
            utils.prepare_tensor("bad_words", bad_words_list, FLAGS.protocol),
            utils.prepare_tensor("stop_words", stop_words_list,
                                 FLAGS.protocol),
        ]

        if FLAGS.protocol == "http":
            async_requests.append(
                client.async_infer(model_name, inputs, request_id=str(i)))
        elif FLAGS.protocol == "grpc":
            async_requests.append(
                client.async_infer(model_name,
                                   inputs,
                                   callback=partial(callback, user_data,
                                                    datetime.now()),
                                   request_id=str(i)))

    if FLAGS.protocol == "http":
        utils.get_http_results(async_requests)
    elif FLAGS.protocol == "grpc":
        utils.get_grpc_results(user_data, len(prompts))
    else:
        raise RuntimeError("Invalid protocol")

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
    parser.add_argument('--max-input-len',
                        type=int,
                        required=True,
                        help='Specify max input length')

    parser.add_argument('--dataset',
                        type=str,
                        required=True,
                        help='Dataset path used for the test.')

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
        print("Encountered error: " + str(e))
        sys.exit(1)

    prompts = []
    output_lens = []
    with open(FLAGS.dataset, 'r') as f:
        data_dict = json.load(f)
        for req in data_dict:
            prompt = req['input'] + ' ' + req['instruction']
            output = req['output']
            # 1.3 is a magic number that converts number of words to number of tokens
            if int(len(prompt.split(' ')) / 1.3) > FLAGS.max_input_len:
                continue
            prompts.append(prompt)
            # 1.3 is a magic number that converts number of words to number of tokens
            output_lens.append(int(len(output.split(' ')) * 1.3))

    test_functionality(client, prompts, output_lens)
    test_performance(client, prompts, output_lens)
