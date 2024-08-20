#!/usr/bin/python

import argparse
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from functools import partial

import numpy as np
import requests
import tritonclient.grpc as grpcclient
from PIL import Image
from transformers import AutoProcessor, Blip2Processor
from utils import utils


def prepare_inputs(text_data, image_data, request_output_len_data,
                   beam_width_data, temperature_data, repetition_penalty_data,
                   presence_penalty_data, end_id, pad_id, top_k_data,
                   top_p_data, streaming_data):
    inputs = [
        utils.prepare_tensor("text_input", text_data, grpcclient),
        utils.prepare_tensor("image_input", image_data, grpcclient),
        utils.prepare_tensor("max_tokens", request_output_len_data,
                             grpcclient),
        utils.prepare_tensor("beam_width", beam_width_data, grpcclient),
        utils.prepare_tensor("temperature", temperature_data, grpcclient),
        utils.prepare_tensor("end_id", end_id, grpcclient),
        utils.prepare_tensor("pad_id", pad_id, grpcclient),
        utils.prepare_tensor("top_k", top_k_data, grpcclient),
        utils.prepare_tensor("top_p", top_p_data, grpcclient),
        utils.prepare_tensor("stream", streaming_data, grpcclient),
    ]
    if repetition_penalty_data is not None:
        inputs += [
            utils.prepare_tensor("repetition_penalty", repetition_penalty_data,
                                 grpcclient),
        ]
    if presence_penalty_data is not None:
        inputs += [
            utils.prepare_tensor("presence_penalty", presence_penalty_data,
                                 grpcclient),
        ]
    return inputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        required=False,
        default=False,
        help="Enable verbose output",
    )
    parser.add_argument(
        "-u",
        "--url",
        type=str,
        required=False,
        default="localhost:8001",
        help="Inference server URL. Default is localhost:8001.",
    )
    parser.add_argument('--text',
                        type=str,
                        required=False,
                        default='Question: which city is this? Answer:',
                        help='Input text')

    parser.add_argument(
        '--image',
        type=str,
        required=False,
        default=
        'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png',
        help='Input image')

    parser.add_argument('--end-id',
                        type=int,
                        required=False,
                        default=50118,
                        help='The token id for end token.')

    parser.add_argument('--pad-id',
                        type=int,
                        required=False,
                        default=1,
                        help='The token id for pad token.')

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
        default=1.0,
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
        "--request-output-len",
        type=int,
        required=False,
        default=16,
        help="Request output length",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        required=False,
        default=1,
        help="top k value",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        required=False,
        default=0.,
        help="top p value",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        required=False,
        default=False,
        help="Enable streaming mode. Default is False.",
    )
    parser.add_argument(
        "--use_bls",
        action="store_true",
        required=False,
        default=False,
        help="Use BLS model instead of ensemble.",
    )
    parser.add_argument("--model_type",
                        required=True,
                        choices=['blip2', 'llava'],
                        help="Model type")
    FLAGS = parser.parse_args()

    raw_image = Image.open(requests.get(FLAGS.image, stream=True).raw)
    if 'blip2' in FLAGS.model_type:
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        image = processor(raw_image, FLAGS.text,
                          return_tensors="pt")['pixel_values']
    elif 'llava' in FLAGS.model_type:
        processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
        image = processor(text=FLAGS.text,
                          images=raw_image,
                          return_tensors="pt")['pixel_values']
    image_data = image.numpy().astype(np.float16)

    text_data = np.array([[FLAGS.text.encode("utf8")]], dtype=np.object_)
    end_id_data = np.array([[FLAGS.end_id]], dtype=np.int32)
    pad_id_data = np.array([[FLAGS.pad_id]], dtype=np.int32)
    request_output_len = [[FLAGS.request_output_len]]
    request_output_len_data = np.array(request_output_len, dtype=np.int32)
    beam_width = [[FLAGS.beam_width]]
    beam_width_data = np.array(beam_width, dtype=np.int32)
    top_k = [[FLAGS.top_k]]
    top_k_data = np.array(top_k, dtype=np.int32)
    top_p = [[FLAGS.top_p]]
    top_p_data = np.array(top_p, dtype=np.float32)
    temperature = [[FLAGS.temperature]]
    temperature_data = np.array(temperature, dtype=np.float32)
    streaming = [[FLAGS.streaming]]
    streaming_data = np.array(streaming, dtype=bool)

    model_name = "ensemble"
    if FLAGS.use_bls:
        model_name = "tensorrt_llm_bls"

    repetition_penalty_data = None
    if FLAGS.repetition_penalty is not None:
        repetition_penalty = [[FLAGS.repetition_penalty]]
        repetition_penalty_data = np.array(repetition_penalty,
                                           dtype=np.float32)
    presence_penalty_data = None
    if FLAGS.presence_penalty is not None:
        presence_penalty = [[FLAGS.presence_penalty]]
        presence_penalty_data = np.array(presence_penalty, dtype=np.float32)

    inputs = prepare_inputs(text_data, image_data, request_output_len_data,
                            beam_width_data, temperature_data,
                            repetition_penalty_data, presence_penalty_data,
                            end_id_data, pad_id_data, top_k_data, top_p_data,
                            streaming_data)

    start_time = datetime.now()

    with utils.create_inference_server_client('grpc',
                                              FLAGS.url,
                                              concurrency=None,
                                              verbose=FLAGS.verbose) as client:
        user_data = utils.UserData()

        if FLAGS.streaming:
            client.start_stream(
                callback=partial(utils.completion_callback, user_data),
                stream_timeout=None,
            )
            client.async_stream_infer(model_name, inputs)
            client.stop_stream(cancel_requests=False)

            results = []
            while True:
                try:
                    (result,
                     error) = user_data._completed_requests.get(block=False)
                    output = result.as_numpy("text_output")
                    for i in range(FLAGS.beam_width):
                        print("[beam", i, "]: ", output[i].decode())
                except Exception:
                    break

        else:
            client.async_infer(model_name, inputs,
                               partial(utils.completion_callback, user_data))
            results = utils.get_grpc_results(user_data, request_parallelism=1)

    stop_time = datetime.now()

    if not FLAGS.streaming:
        output = results[0].as_numpy("text_output")
        for i in range(FLAGS.beam_width):
            print("[beam", i, "]:")
            print(output[i].decode())

    latency = (stop_time - start_time).total_seconds() * 1000.0
    latency = round(latency, 3)
    print(f"[INFO] Latency: {latency} ms")
