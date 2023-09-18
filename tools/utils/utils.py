import queue
from functools import partial

import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype


class UserData:

    def __init__(self):
        self._completed_requests = queue.Queue()
        self._latencies = []


# Callback function used for async_stream_infer()
def completion_callback(user_data, result, error):
    # passing error raise and handling out
    user_data._completed_requests.put((result, error))


def prepare_tensor(name, input, protocol):
    client_util = httpclient if protocol == "http" else grpcclient
    t = client_util.InferInput(name, input.shape,
                               np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t


def prepare_inputs(input_start_ids, input_len, pad_id, end_id, flags):
    output_len = np.ones([input_start_ids.shape[0], 1]).astype(
        np.uint32) * flags.output_len
    runtime_top_k = (flags.topk *
                     np.ones([input_start_ids.shape[0], 1])).astype(np.uint32)
    runtime_top_p = flags.topp * \
        np.ones([input_start_ids.shape[0], 1]).astype(np.float32)
    beam_search_diversity_rate = 0.0 * \
        np.ones([input_start_ids.shape[0], 1]).astype(np.float32)
    temperature = 1.0 * \
        np.ones([input_start_ids.shape[0], 1]).astype(np.float32)
    len_penalty = 1.0 * \
        np.ones([input_start_ids.shape[0], 1]).astype(np.float32)
    repetition_penalty = 1.0 * \
        np.ones([input_start_ids.shape[0], 1]).astype(np.float32)
    random_seed = 0 * \
        np.ones([input_start_ids.shape[0], 1]).astype(np.uint64)
    output_log_probs = True * \
        np.ones([input_start_ids.shape[0], 1]).astype(bool)
    beam_width = (flags.beam_width *
                  np.ones([input_start_ids.shape[0], 1])).astype(np.uint32)
    pad_ids = pad_id * \
        np.ones([input_start_ids.shape[0], 1]).astype(np.uint32)
    end_ids = end_id * \
        np.ones([input_start_ids.shape[0], 1]).astype(np.uint32)
    min_length = 1 * \
        np.ones([input_start_ids.shape[0], 1]).astype(np.uint32)
    presence_penalty = 0.0 * \
        np.ones([input_start_ids.shape[0], 1]).astype(np.float32)
    bad_words_list = np.concatenate([
        np.zeros([input_start_ids.shape[0], 1, 1]).astype(np.int32),
        (-1 * np.ones([input_start_ids.shape[0], 1, 1])).astype(np.int32)
    ],
                                    axis=1)
    stop_word_list = np.concatenate([
        np.zeros([input_start_ids.shape[0], 1, 1]).astype(np.int32),
        (-1 * np.ones([input_start_ids.shape[0], 1, 1])).astype(np.int32)
    ],
                                    axis=1)
    inputs = [
        prepare_tensor("input_ids", input_start_ids, flags.protocol),
        prepare_tensor("input_lengths", input_len, flags.protocol),
        prepare_tensor("request_output_len", output_len, flags.protocol),
        prepare_tensor("pad_id", pad_ids, flags.protocol),
        prepare_tensor("end_id", end_ids, flags.protocol),
        prepare_tensor("beam_width", beam_width, flags.protocol),
        prepare_tensor("temperature", temperature, flags.protocol),
        prepare_tensor("runtime_top_k", runtime_top_k, flags.protocol),
        prepare_tensor("runtime_top_p", runtime_top_p, flags.protocol),
        prepare_tensor("len_penalty", len_penalty, flags.protocol),
        prepare_tensor("repetition_penalty", repetition_penalty,
                       flags.protocol),
        prepare_tensor("min_length", min_length, flags.protocol),
        prepare_tensor("presence_penalty", presence_penalty, flags.protocol),
        prepare_tensor("random_seed", random_seed, flags.protocol),
        prepare_tensor("output_log_probs", output_log_probs, flags.protocol),
        # prepare_tensor("bad_words_list", bad_words_list, flags.protocol),
        # prepare_tensor("stop_words_list", stop_word_list, flags.protocol),
    ]
    return inputs


def create_inference_server_client(protocol, url, concurrency, verbose):
    client_util = httpclient if protocol == "http" else grpcclient
    if protocol == "http":
        return client_util.InferenceServerClient(url,
                                                 concurrency=concurrency,
                                                 verbose=verbose)
    elif protocol == "grpc":
        return client_util.InferenceServerClient(url, verbose=verbose)


def send_requests(model_name, inputs, client, request_parallelism):
    results = []
    for _ in range(request_parallelism):
        result = client.infer(model_name, inputs)
        results.append(result)
    return results


def send_requests_async(model_name, inputs, client, flags,
                        request_parallelism):
    if flags.protocol == "http":
        async_requests = []
        for _ in range(request_parallelism):
            async_requests.append(client.async_infer(model_name, inputs))
        return async_requests
    else:
        user_data = UserData()
        for _ in range(request_parallelism):
            client.async_infer(model_name, inputs,
                               partial(completion_callback, user_data))
        return user_data


def get_http_results(async_requests):
    results = []
    for async_request in async_requests:
        results.append(async_request.get_result())
    return results


def get_grpc_results(user_data, request_parallelism):
    results = []
    processed_count = 0
    while processed_count < request_parallelism:
        (result, error) = user_data._completed_requests.get()
        processed_count += 1
        if error is not None:
            raise RuntimeError(error)
        results.append(result)
    return results


def append_start_and_end_ids(inputs,
                             batch_size,
                             flags,
                             start_id=None,
                             end_id=None):
    if start_id is not None:
        start_ids = start_id * np.ones([batch_size, 1]).astype(np.uint32)
        inputs.append(prepare_tensor("start_id", start_ids, flags.protocol))
    if end_id is not None:
        end_ids = end_id * np.ones([batch_size, 1]).astype(np.uint32)
        inputs.append(prepare_tensor("end_id", end_ids, flags.protocol))
