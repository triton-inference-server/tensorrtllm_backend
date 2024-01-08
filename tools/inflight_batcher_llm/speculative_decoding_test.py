#!/usr/bin/python

import os
import sys

utils_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
root_path = os.path.dirname(utils_path)
sys.path.append(utils_path)
sys.path.append(os.path.join(root_path, "inflight_batcher_llm"))

import argparse
import json
import sys

import tritonclient.grpc as grpcclient
from client import e2e_grpc_speculative_decoding_client, end_to_end_grpc_client

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')

    parser.add_argument('--url-target',
                        type=str,
                        required=True,
                        help='Inference server URL for the target model')

    parser.add_argument('--url-draft',
                        type=str,
                        required=True,
                        help='Inference server URL for the draft model')

    parser.add_argument('--max-input-len',
                        type=int,
                        required=True,
                        help='Max input length for input prompts')

    parser.add_argument(
        '--preprocessor-model-name',
        type=str,
        required=False,
        default="preprocessing",
        help='Name of the preprocessor model (should be hosted at url-draft)')

    parser.add_argument(
        '--postprocessor-model-name',
        type=str,
        required=False,
        default="postprocessing",
        help='Name of the postprocessor model (should be hosted at url-target)'
    )

    parser.add_argument(
        '--draft-tensorrt-llm-model-name',
        type=str,
        required=False,
        default="tensorrt_llm",
        help='Name of the tensorrt_llm draft model (hosted at url-draft)')

    parser.add_argument(
        '--target-tensorrt-llm-model-name',
        type=str,
        required=False,
        default="tensorrt_llm",
        help='Name of the tensorrt_llm draft model (hosted at url-draft)')

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
        "--frequency-penalty",
        type=float,
        required=False,
        default=None,
        help="The frequency penalty value",
    )

    parser.add_argument('-o',
                        '--output-len',
                        type=int,
                        default=100,
                        required=False,
                        help='Specify output length')

    parser.add_argument(
        '--num-draft-tokens',
        type=int,
        default=5,
        required=False,
        help=
        'Specify the number of speculative tokens for the draft model to generate per lookahead.'
    )

    parser.add_argument('--end-id',
                        type=int,
                        default=None,
                        required=False,
                        help='The end if token')

    parser.add_argument('--pad-id',
                        type=int,
                        default=None,
                        required=False,
                        help='The pad if token')

    parser.add_argument('--stop-words',
                        nargs='+',
                        default=[],
                        help='The stop words')

    parser.add_argument('--bad-words',
                        nargs='+',
                        default=[],
                        help='The bad words')

    parser.add_argument('--dataset',
                        type=str,
                        required=True,
                        help='Dataset path used for the test.')

    FLAGS = parser.parse_args()
    if not FLAGS.url_target:
        FLAGS.url_target = "localhost:8001"

    if not FLAGS.url_draft:
        FLAGS.url_draft = FLAGS.url_target

    try:
        client_target = grpcclient.InferenceServerClient(url=FLAGS.url_target)
        client_draft = grpcclient.InferenceServerClient(
            url=FLAGS.url_draft) if (
                FLAGS.url_target != FLAGS.url_draft) else client_target
    except Exception as e:
        print("client creation failed: " + str(e))
        sys.exit(1)

    if (FLAGS.beam_width > 1):
        raise Exception(
            'Beam width > 1 is not yet supported with speculative decoding')

    request_id = 1
    total_count = 0
    failed_count = 0
    with open(FLAGS.dataset, 'r') as f:
        data_dict = json.load(f)
        for req in data_dict:
            prompt = req['input'] + ' ' + req['instruction']
            output = req['output']
            # 1.3 is a magic number that converts number of words to number of tokens
            if int(len(prompt.split(' ')) / 1.3) > FLAGS.max_input_len:
                continue
            # 1.3 is a magic number that converts number of words to number of tokens
            output_len = int(len(output.split(' ')) * 1.3)
            if FLAGS.verbose:
                print(f"Prompt: {prompt}")
                print(f"Output len: {output_len}")

            # Calling target model only
            if FLAGS.verbose:
                print(f"Calling target", flush=True)
            output_target = end_to_end_grpc_client.run_inference(
                client_target, prompt, output_len, str(request_id),
                FLAGS.repetition_penalty, FLAGS.presence_penalty,
                FLAGS.frequency_penalty, FLAGS.temperature, FLAGS.stop_words,
                FLAGS.bad_words, [], [], "ensemble", False, 1, False, None,
                None, FLAGS.verbose)
            if FLAGS.verbose:
                print(f"output_target: {output_target}", flush=True)

            if FLAGS.verbose:
                print(f"Calling speculative", flush=True)
            output_speculative = e2e_grpc_speculative_decoding_client.run_speculative_inference(
                client_draft,
                client_target, prompt, output_len, FLAGS.num_draft_tokens,
                str(request_id), FLAGS.repetition_penalty,
                FLAGS.presence_penalty, FLAGS.frequency_penalty,
                FLAGS.temperature, FLAGS.stop_words, FLAGS.bad_words,
                FLAGS.end_id, FLAGS.pad_id, FLAGS.beam_width,
                FLAGS.preprocessor_model_name,
                FLAGS.draft_tensorrt_llm_model_name,
                FLAGS.target_tensorrt_llm_model_name,
                FLAGS.postprocessor_model_name, FLAGS.verbose)
            if FLAGS.verbose:
                print(f"output_speculative: {output_speculative}", flush=True)

            total_count = total_count + 1
            if (output_target != output_speculative):
                failed_count = failed_count + 1
                print(f"{total_count}: Outputs don't match")
                print(f"Prompt:")
                print(f"{prompt}")
                print(f"Output target:")
                print(f"{output_target}")
                print(f"Output speculative:")
                print(f"{output_speculative}")
            else:
                print(f"{total_count}: Outputs match")
            request_id = request_id + 1

    print(f"failed/total: {failed_count}/{total_count}")
    sys.exit(failed_count > 0)
