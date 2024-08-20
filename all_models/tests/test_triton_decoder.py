# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import sys
from dataclasses import dataclass
from typing import Dict, List, Union
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# Mock pb_utils
sys.modules["triton_python_backend_utils"] = MagicMock()

from lib.decode import GenerationResponse, PreprocResponse, Request, Response
# Use PYTHONPATH=../inflight_batcher_llm/tensorrt_llm_bls/1/
from lib.triton_decoder import TritonDecoder


@dataclass
class MockTritonTensor:
    _name: str
    _tensor: Union[np.ndarray, torch.Tensor]

    def name(self) -> str:
        return self._name

    def as_numpy(self) -> np.ndarray:
        if self.is_cpu():
            return self._tensor
        else:
            return self._tensor.as_numpy()

    def is_cpu(self) -> bool:
        if isinstance(self._tensor, np.ndarray):
            return True
        else:
            return False


@dataclass
class MockTritonResponse:
    tensors: Dict[str, MockTritonTensor]

    def __init__(self, output_tensors: List[MockTritonTensor]):
        self.tensors = {}
        for tensor in output_tensors:
            self.tensors[tensor.name()] = tensor

    def output_tensors(self):
        return self.tensors.values()


@dataclass
class MockTritonRequest:
    tensors: Dict[str, MockTritonTensor]

    def get_input_tensor_by_name(self, name: str) -> MockTritonTensor:
        return self.tensors[name] if name in self.tensors else None

    def get_response_sender(self):
        return None


@pytest.fixture
def triton_decoder() -> TritonDecoder:
    return TritonDecoder()


@pytest.fixture
def response(request) -> MockTritonResponse:
    output_names = [
        "text_output",
        "cum_log_probs",
        "output_log_probs",
        "context_logits",
        "generation_logits",
        "batch_index",
    ]
    response = Response()
    for output_name in output_names:
        setattr(response, output_name, np.array(request.param[output_name]))
    return response


@pytest.fixture
def triton_request(request) -> MockTritonRequest:
    input_names = [
        "text_input", "max_tokens", "bad_words", "stop_words", "end_id",
        "pad_id", "top_k", "top_p", "temperature", "length_penalty",
        "repetition_penalty", "min_length", "presence_penalty",
        "frequency_penalty", "random_seed", "return_log_probs",
        "return_context_logits", "return_generation_logits", "beam_width",
        "stream", "prompt_embedding_table", "prompt_vocab_size",
        "embedding_bias_words", "embedding_bias_weights", "num_draft_tokens"
    ]
    triton_tensor_map = {}
    for input_name in input_names:
        if input_name in request.param:
            triton_tensor = MockTritonTensor(
                input_name, np.array(request.param[input_name]))
            triton_tensor_map[input_name] = triton_tensor
    return MockTritonRequest(triton_tensor_map)


@pytest.fixture(autouse=True)
def apply_patches():
    patch("lib.triton_decoder.pb_utils.Tensor", new=MockTritonTensor).start()
    patch("lib.triton_decoder.pb_utils.InferenceResponse",
          new=MockTritonResponse).start()
    patch("lib.triton_decoder.pb_utils.InferenceRequest",
          new=MockTritonRequest).start()
    patch("lib.triton_decoder.pb_utils.get_input_tensor_by_name",
          new=mock_pb_utils_get_input_tensor_by_name_side_effect).start()


def mock_pb_utils_get_input_tensor_by_name_side_effect(
        request: MockTritonRequest, name: str) -> MockTritonTensor:
    return request.get_input_tensor_by_name(name)


mock_reponse = {
    "text_output": ["Hello world"],
    "cum_log_probs": [[0.0]],
    "output_log_probs": [[[0.1, 0.3]]],
    "context_logits": [[[-0.2, 0.2]]],
    "generation_logits": [[[0.3, 1.1]]],
    "batch_index": [[0]]
}

mock_request = {"text_input": [["Hello world"]], "max_tokens": [[24]]}


@pytest.mark.parametrize("response", [mock_reponse], indirect=True)
def test_create_triton_response(triton_decoder: TritonDecoder,
                                response: Response):
    triton_response = triton_decoder.create_triton_response(response)
    # Check if all fields and values are present in the triton response
    output_triton_tensors = triton_response.output_tensors()
    output_triton_tensor_map = {
        tensor.name(): tensor.as_numpy()
        for tensor in output_triton_tensors
    }
    assert (output_triton_tensor_map.keys() == response.__dict__.keys())
    for output_name in output_triton_tensor_map:
        output_tensor = output_triton_tensor_map[output_name]
        np.testing.assert_array_equal(output_tensor,
                                      getattr(response, output_name))


@pytest.mark.parametrize("triton_request", [mock_request], indirect=True)
def test_convert_triton_request(triton_decoder: TritonDecoder,
                                triton_request: MockTritonRequest):
    request = triton_decoder.convert_triton_request(triton_request)
    tensor_names = [
        tensor_name for tensor_name in request.__dict__.keys()
        if getattr(request, tensor_name) is not None
    ]
    assert set(tensor_names) == triton_request.tensors.keys()
    for tensor_name in tensor_names:
        request_tensor = getattr(request, tensor_name)
        if request_tensor is not None:
            triton_tensor = triton_request.get_input_tensor_by_name(
                tensor_name)
            assert triton_tensor is not None
            np.testing.assert_array_equal(getattr(request, tensor_name),
                                          triton_tensor.as_numpy())


_preproc_name_map = {
    "INPUT_ID": "input_ids",
    "REQUEST_INPUT_LEN": "input_lengths",
    "BAD_WORDS_IDS": "bad_words_list",
    "STOP_WORDS_IDS": "stop_words_list",
    "EMBEDDING_BIAS": "embedding_bias",
    "OUT_PAD_ID": "pad_id",
    "OUT_END_ID": "end_id",
}
_generation_name_map = {
    "output_ids": "output_ids",
    "sequence_length": "sequence_length",
    "cum_log_probs": "cum_log_probs",
    "output_log_probs": "output_log_probs",
    "context_logits": "context_logits",
    "generation_logits": "generation_logits",
    "batch_index": "batch_index"
}

convert_triton_response_testcases = [{
    "response_factory": PreprocResponse,
    "name_map": _preproc_name_map,
    "response": {
        "INPUT_ID": [["Hello world"]],
        "REQUEST_INPUT_LEN": [[16]]
    }
}, {
    "response_factory": GenerationResponse,
    "name_map": _generation_name_map,
    "response": {
        "output_ids": [[[1, 23, 23412, 2]]],
        "sequence_length": [[4]]
    }
}]


@pytest.mark.parametrize("convert_triton_response_testcases",
                         convert_triton_response_testcases)
def test_convert_triton_response(triton_decoder: TritonDecoder,
                                 convert_triton_response_testcases):
    triton_tensors = []
    for tensor_name, tensor in convert_triton_response_testcases[
            "response"].items():
        triton_tensors.append(MockTritonTensor(tensor_name, np.array(tensor)))
    triton_response = MockTritonResponse(triton_tensors)
    response = triton_decoder.convert_triton_response(
        triton_response, convert_triton_response_testcases["response_factory"],
        convert_triton_response_testcases["name_map"])

    response_tensors_length = len([
        attr for attr in response.__dict__
        if getattr(response, attr) is not None
    ])
    assert len(convert_triton_response_testcases["response"]
               ) == response_tensors_length
    for tensor_name, tensor in convert_triton_response_testcases[
            "response"].items():
        target_name = tensor_name
        if convert_triton_response_testcases["name_map"]:
            target_name = convert_triton_response_testcases["name_map"][
                tensor_name]
        assert getattr(response, target_name) is not None
        np.testing.assert_array_equal(
            convert_triton_response_testcases["response"][tensor_name],
            getattr(response, target_name))


create_triton_tensors_testcases = [{
    "obj":
    Request(text_input=np.array([["Hello world"]]),
            max_tokens=np.array([["16"]]),
            return_log_probs=np.array([True])),
    "name_map": {
        "text_input": "QUERY",
        "max_tokens": "REQUEST_OUTPUT_LEN",
        "return_log_probs": "return_log_probs",
    },
    "undo_reshape_map": {
        "return_log_probs": True,
    }
}]


@pytest.mark.parametrize("create_triton_tensors_testcases",
                         create_triton_tensors_testcases)
def test_create_triton_tensors(triton_decoder: TritonDecoder,
                               create_triton_tensors_testcases):
    request = create_triton_tensors_testcases["obj"]
    obj_tensors_length = len([
        attr for attr in request.__dict__ if getattr(request, attr) is not None
    ])
    triton_tensors = triton_decoder.create_triton_tensors(
        create_triton_tensors_testcases["obj"],
        create_triton_tensors_testcases["name_map"])
    triton_tensor_map = {
        tensor.name(): tensor.as_numpy()
        for tensor in triton_tensors
    }
    assert len(triton_tensors) == obj_tensors_length
    for tensor_name in request.__dict__:
        if getattr(request, tensor_name) is not None:
            target_name = create_triton_tensors_testcases["name_map"][
                tensor_name]
            assert target_name in triton_tensor_map
            if create_triton_tensors_testcases.get("undo_reshape_map",
                                                   {}).get(target_name, False):
                np.testing.assert_array_equal(
                    triton_tensor_map[target_name],
                    np.expand_dims(getattr(request, tensor_name), 0))
            else:
                np.testing.assert_array_equal(triton_tensor_map[target_name],
                                              getattr(request, tensor_name))
