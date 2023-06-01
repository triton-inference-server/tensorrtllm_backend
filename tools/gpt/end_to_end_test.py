#!/usr/bin/python

import argparse

import numpy as np
from utils import utils

# GPT3 Related variables
# Reference : https://github.com/NVIDIA/FasterTransformer/blob/main/sample/pytorch/gpt_sample.py
MERGES_FILE = "gpt2-merges.txt"
VOCAB_FILE = "gpt2-vocab.json"

GPT_START_ID = 220
GPT_END_ID = 50256

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

    model_name = 'preprocessing'
    with utils.create_inference_server_client(FLAGS.protocol,
                                              FLAGS.url,
                                              concurrency=FLAGS.concurrency,
                                              verbose=FLAGS.verbose) as client:
        input0 = [["Blackhawks\n The 2015 Hilltoppers"],
                  ["Data sources you can use to make a decision:"],
                  ["\n if(angle = 0) { if(angle"],
                  ["GMs typically get 78% female enrollment, but the "],
                  ["Previous Chapter | Index | Next Chapter"],
                  ["Michael, an American Jew, called Jews"],
                  ["Born in north-east France, Soyer trained as a"],
                  ["Data sources you can use to make a comparison:"]]
        input0_data = np.array(input0).astype(object)
        output0_len = np.ones_like(input0).astype(np.uint32) * FLAGS.output_len
        bad_words_list = np.array(
            [["Hawks, Hawks"], [""], [""], [""], [""], [""], [""], [""]],
            dtype=object)
        stop_words_list = np.array(
            [[""], [""], [""], [""], [""], [""], [""], ["month, month"]],
            dtype=object)
        inputs = [
            utils.prepare_tensor("QUERY", input0_data, FLAGS.protocol),
            utils.prepare_tensor("BAD_WORDS_DICT", bad_words_list,
                                 FLAGS.protocol),
            utils.prepare_tensor("STOP_WORDS_DICT", stop_words_list,
                                 FLAGS.protocol),
            utils.prepare_tensor("REQUEST_OUTPUT_LEN", output0_len,
                                 FLAGS.protocol),
        ]

        try:
            result = client.infer(model_name, inputs)
            output0 = result.as_numpy("INPUT_ID")
            output1 = result.as_numpy("REQUEST_INPUT_LEN")
            output2 = result.as_numpy("REQUEST_OUTPUT_LEN")
            output3 = result.as_numpy("BAD_WORDS_IDS")
            output4 = result.as_numpy("STOP_WORDS_IDS")
        except Exception as e:
            print(e)

    model_name = "tensorrt_llm"
    with utils.create_inference_server_client(FLAGS.protocol,
                                              FLAGS.url,
                                              concurrency=1,
                                              verbose=FLAGS.verbose) as client:
        inputs = utils.prepare_inputs(output0, output1, FLAGS)

        try:
            result = client.infer(model_name, inputs)
            output0 = result.as_numpy("output_ids")
        except Exception as e:
            print(e)

    model_name = "postprocessing"
    with utils.create_inference_server_client(FLAGS.protocol,
                                              FLAGS.url,
                                              concurrency=FLAGS.concurrency,
                                              verbose=FLAGS.verbose) as client:
        inputs = [
            utils.prepare_tensor("TOKENS_BATCH", output0, FLAGS.protocol)
        ]
        inputs[0].set_data_from_numpy(output0)

        try:
            result = client.infer(model_name, inputs)
            output0 = result.as_numpy("OUTPUT")
            print("============After postprocessing============")
            batch_size = len(input0)
            output0 = output0.reshape([-1, batch_size]).T.tolist()
            output0 = [[char.decode('UTF-8') for char in line]
                       for line in output0]
            output0 = [''.join(line) for line in output0]
            for line in output0:
                print(f"{line}")
            print("===========================================\n\n\n")
        except Exception as e:
            print(e)

    model_name = "ensemble"
    with utils.create_inference_server_client(FLAGS.protocol,
                                              FLAGS.url,
                                              concurrency=FLAGS.concurrency,
                                              verbose=FLAGS.verbose) as client:
        input0 = [["Blackhawks\n The 2015 Hilltoppers"],
                  ["Data sources you can use to make a decision:"],
                  ["\n if(angle = 0) { if(angle"],
                  ["GMs typically get 78% female enrollment, but the "],
                  ["Previous Chapter | Index | Next Chapter"],
                  ["Michael, an American Jew, called Jews"],
                  ["Born in north-east France, Soyer trained as a"],
                  ["Data sources you can use to make a comparison:"]]
        bad_words_list = np.array(
            [["Hawks, Hawks"], [""], [""], [""], [""], [""], [""], [""]],
            dtype=object)
        stop_words_list = np.array(
            [[""], [""], [""], [""], [""], [""], [""], ["month, month"]],
            dtype=object)
        input0_data = np.array(input0).astype(object)
        output0_len = np.ones_like(input0).astype(np.uint32) * FLAGS.output_len
        runtime_top_k = (FLAGS.topk *
                         np.ones([input0_data.shape[0], 1])).astype(np.uint32)
        runtime_top_p = FLAGS.topp * np.ones([input0_data.shape[0], 1]).astype(
            np.float32)
        beam_search_diversity_rate = 0.0 * np.ones([input0_data.shape[0], 1
                                                    ]).astype(np.float32)
        temperature = 1.0 * np.ones([input0_data.shape[0], 1]).astype(
            np.float32)
        len_penalty = 1.0 * np.ones([input0_data.shape[0], 1]).astype(
            np.float32)
        repetition_penalty = 1.0 * np.ones([input0_data.shape[0], 1]).astype(
            np.float32)
        random_seed = 0 * np.ones([input0_data.shape[0], 1]).astype(np.uint64)
        is_return_log_probs = True * np.ones([input0_data.shape[0], 1
                                              ]).astype(bool)
        beam_width = (FLAGS.beam_width *
                      np.ones([input0_data.shape[0], 1])).astype(np.uint32)
        inputs = [
            utils.prepare_tensor("INPUT_0", input0_data, FLAGS.protocol),
            utils.prepare_tensor("INPUT_1", output0_len, FLAGS.protocol),
            utils.prepare_tensor("INPUT_2", bad_words_list, FLAGS.protocol),
            utils.prepare_tensor("INPUT_3", stop_words_list, FLAGS.protocol),
            utils.prepare_tensor("runtime_top_k", runtime_top_k,
                                 FLAGS.protocol),
            utils.prepare_tensor("runtime_top_p", runtime_top_p,
                                 FLAGS.protocol),
            utils.prepare_tensor("beam_search_diversity_rate",
                                 beam_search_diversity_rate, FLAGS.protocol),
            utils.prepare_tensor("temperature", temperature, FLAGS.protocol),
            utils.prepare_tensor("len_penalty", len_penalty, FLAGS.protocol),
            utils.prepare_tensor("repetition_penalty", repetition_penalty,
                                 FLAGS.protocol),
            utils.prepare_tensor("random_seed", random_seed, FLAGS.protocol),
            utils.prepare_tensor("beam_width", beam_width, FLAGS.protocol),
        ]

        try:
            result = client.infer(model_name, inputs)
            ensemble_output0 = result.as_numpy("OUTPUT_0")
            print("============After ensemble============")
            batch_size = len(input0)
            ensemble_output0 = ensemble_output0.reshape([-1, batch_size
                                                         ]).T.tolist()
            ensemble_output0 = [[char.decode('UTF-8') for char in line]
                                for line in ensemble_output0]
            ensemble_output0 = [''.join(line) for line in ensemble_output0]
            for line in ensemble_output0:
                print(f"{line}")
        except Exception as e:
            print(e)

    assert output0 == ensemble_output0
