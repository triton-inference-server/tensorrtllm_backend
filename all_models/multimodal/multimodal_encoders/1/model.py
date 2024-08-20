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

import json
import os

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import from_dlpack, to_dlpack

import tensorrt_llm
from tensorrt_llm._utils import str_dtype_to_torch, str_dtype_to_trt
from tensorrt_llm.runtime import Session, TensorInfo


def triton_string_to_torch(dtype):
    type_map = {
        "TYPE_BOOL": torch.bool,
        "TYPE_UINT8": torch.uint8,
        "TYPE_INT8": torch.int8,
        "TYPE_INT16": torch.int16,
        "TYPE_INT32": torch.int32,
        "TYPE_INT64": torch.int64,
        "TYPE_FP16": torch.float16,
        "TYPE_FP32": torch.float32,
        "TYPE_FP64": torch.float64,
    }
    return type_map[dtype]


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        # Will load non-llm engiens only to GPU0 since the requests are coming only to GPU0

        self.rank = tensorrt_llm.mpi_rank()
        if self.rank != 0:
            return

        # Parse model configs
        model_config = json.loads(args['model_config'])

        # First vision engine
        visual_model_path = model_config['parameters'].get(
            'visual_model_path', None)
        if visual_model_path:
            visual_model_path = visual_model_path['string_value']
            self.vision_stream = torch.cuda.current_stream()
            visual_encoder_path = os.path.join(visual_model_path,
                                               'model.engine')
            with open(visual_encoder_path, 'rb') as f:
                engine_buffer = f.read()
            self.image_session = Session.from_serialized_engine(engine_buffer)

            visual_config_path = os.path.join(visual_model_path, 'config.json')
            with open(visual_config_path, 'r') as f:
                visual_config = json.load(f)

            self.vision_dtype_str = visual_config['builder_config'][
                'precision']
            self.vision_output_dtype = triton_string_to_torch(
                pb_utils.get_output_config_by_name(
                    model_config, "OUT_PROMPT_EMBEDDING_TABLE")['data_type'])

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse.
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        pb_utils.Logger
        for idx, request in enumerate(requests):
            # Get input tensors
            img_tensor = pb_utils.get_input_tensor_by_name(request, 'IMAGE')
            img_tensor = from_dlpack(img_tensor.to_dlpack()).pin_memory()
            vit_output_info = self.image_session.infer_shapes([
                TensorInfo('input', str_dtype_to_trt(self.vision_dtype_str),
                           img_tensor.shape)
            ])
            self.vit_output = {
                t.name:
                torch.empty(tuple(t.shape),
                            dtype=str_dtype_to_torch(self.vision_dtype_str),
                            device='cuda')
                for t in vit_output_info
            }
            vit_input = {'input': img_tensor}
            with torch.cuda.stream(self.vision_stream):
                img_tensor = img_tensor.to('cuda')
                self.image_session.run(vit_input, self.vit_output,
                                       self.vision_stream.cuda_stream)
                vision_prompt_table = self.vit_output['output'].to(
                    self.vision_output_dtype)
            self.vision_stream.synchronize()
            vision_prompt_vocab_size = np.array(
                [[vision_prompt_table.shape[1]]])

            # NOTE
            # User can concat the prompt table and prompt vocab size after another session
            prompt_table = vision_prompt_table
            prompt_vocab_size = vision_prompt_vocab_size

            prompt_embedding_table_tensor = pb_utils.Tensor.from_dlpack(
                'OUT_PROMPT_EMBEDDING_TABLE', to_dlpack(prompt_table))

            prompt_vocab_size_tensor = pb_utils.Tensor(
                'OUT_PROMPT_VOCAB_SIZE', prompt_vocab_size.astype(np.int32))

            inference_response = pb_utils.InferenceResponse(output_tensors=[
                prompt_embedding_table_tensor, prompt_vocab_size_tensor
            ])
            responses.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
