import json

import numpy as np


def add_data(data, name, array):
    data['data'][0][name] = {
        'content': array.flatten().tolist(),
        'shape': array.shape
    }


def main():
    data = {'data': [{}]}

    start_len = 20
    output_len = 10
    topk = 1
    topp = 0.0
    beam_width = 1

    input_start_ids = np.random.randint(0,
                                        50255,
                                        size=(start_len),
                                        dtype=np.int32)
    output_len = np.ones([1]).astype(np.uint32) * output_len
    runtime_top_k = (topk * np.ones([1])).astype(np.uint32)
    runtime_top_p = topp * np.ones([1]).astype(np.float32)
    beam_search_diversity_rate = 0.0 * np.ones([1]).astype(np.float32)
    temperature = 1.0 * np.ones([1]).astype(np.float32)
    len_penalty = 1.0 * np.ones([1]).astype(np.float32)
    repetition_penalty = 1.0 * np.ones([1]).astype(np.float32)
    random_seed = 0 * np.ones([1]).astype(np.uint64)
    True * np.ones([1]).astype(bool)
    beam_width = (beam_width * np.ones([1])).astype(np.uint32)
    50256 * np.ones([1]).astype(np.uint32)
    50256 * np.ones([1]).astype(np.uint32)
    bad_words_list = np.concatenate([
        np.zeros([1, 1]).astype(np.int32),
        (-1 * np.ones([1, 1])).astype(np.int32)
    ],
                                    axis=1)
    stop_word_list = np.concatenate([
        np.zeros([1, 1]).astype(np.int32),
        (-1 * np.ones([1, 1])).astype(np.int32)
    ],
                                    axis=1)

    add_data(data, 'input_ids', input_start_ids)
    add_data(data, 'request_output_len', output_len)
    add_data(data, 'runtime_top_k', runtime_top_k)
    add_data(data, 'runtime_top_p', runtime_top_p)
    add_data(data, 'beam_search_diversity_rate', beam_search_diversity_rate)
    add_data(data, 'temperature', temperature)
    add_data(data, 'len_penalty', len_penalty)
    add_data(data, 'repetition_penalty', repetition_penalty)
    add_data(data, 'random_seed', random_seed)
    add_data(data, 'beam_width', beam_width)
    # add_data(data, 'top_p_decay', top_p_decay)
    # add_data(data, 'top_p_min', top_p_min)
    # add_data(data, 'top_p_reset_ids', top_p_reset_ids)

    with open('input_data.json', 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == '__main__':
    main()
