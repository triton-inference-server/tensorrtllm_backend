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
import tensorrt_llm.logger as logger
from tensorrt_llm._utils import str_dtype_to_torch, torch_dtype_to_trt
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
        "TYPE_BF16": torch.bfloat16
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

            visual_config_path = os.path.join(visual_model_path, 'config.json')
            with open(visual_config_path, 'r') as f:
                visual_config = json.load(f)
            self.model_type = visual_config['builder_config']['model_type']

            visual_encoder_path = os.path.join(
                visual_model_path, 'model.engine'
                if self.model_type != "mllama" else 'visual_encoder.engine')
            with open(visual_encoder_path, 'rb') as f:
                engine_buffer = f.read()
            self.image_session = Session.from_serialized_engine(engine_buffer)

            self.vision_dtype_str = visual_config['builder_config'][
                'precision']
            features_output_name = "OUT_PROMPT_EMBEDDING_TABLE"
            if self.model_type == "mllama":
                features_output_name = "ENCODER_INPUT_FEATURES"
            self.vision_output_dtype = triton_string_to_torch(
                pb_utils.get_output_config_by_name(
                    model_config, features_output_name)['data_type'])

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
            img_tensor = (
                pb_utils.get_input_tensor_by_name(request, 'pixel_values')
                or pb_utils.get_input_tensor_by_name(request, 'IMAGE'))
            assert img_tensor != None, "There is no preprocessed image tensor to encode"
            img_tensor = from_dlpack(img_tensor.to_dlpack())

            batch_size = img_tensor.shape[0]
            num_images = img_tensor.shape[1]
            img_tensor = img_tensor.to(
                str_dtype_to_torch(self.vision_dtype_str)).pin_memory()

            # Prepare input tensors for the vision encoder
            if self.model_type != 'mllama':
                vit_input = {
                    'input':
                    img_tensor.view(batch_size * num_images,
                                    img_tensor.shape[2], img_tensor.shape[3],
                                    img_tensor.shape[4])
                }
            else:
                aspect_ratio_ids = from_dlpack(
                    pb_utils.get_input_tensor_by_name(
                        request, "aspect_ratio_ids").to_dlpack()).to(
                            torch.int64).pin_memory()
                aspect_ratio_mask = from_dlpack(
                    pb_utils.get_input_tensor_by_name(
                        request, "aspect_ratio_mask").to_dlpack()).to(
                            torch.int64).pin_memory()
                num_tiles = aspect_ratio_mask.shape[-1]
                # reshape img_tensor to [bs, num_images, num_tiles, ...]
                pixel_values = img_tensor.view(img_tensor.shape[0], -1,
                                               num_tiles,
                                               *(img_tensor.shape[2:]))
                vit_input = {
                    'pixel_values': pixel_values,
                    'aspect_ratio_ids': aspect_ratio_ids,
                    'aspect_ratio_mask': aspect_ratio_mask
                }

            # Run the vision encoder
            vit_input_info = [
                TensorInfo(key, torch_dtype_to_trt(val.dtype), val.shape)
                for key, val in vit_input.items()
            ]
            vit_output_info = self.image_session.infer_shapes(vit_input_info)
            vit_output = {
                t.name:
                torch.empty(tuple(t.shape),
                            dtype=str_dtype_to_torch(self.vision_dtype_str),
                            device='cuda')
                for t in vit_output_info
            }
            with torch.cuda.stream(self.vision_stream):
                img_tensor = img_tensor.to('cuda')
                ok = self.image_session.run(vit_input, vit_output,
                                            self.vision_stream.cuda_stream)
                assert ok, "Runtime execution failed for vision encoder session"
                embeddings = vit_output['output'].to(self.vision_output_dtype)
            self.vision_stream.synchronize()

            output_tensors = []

            # Create output tensors
            if self.model_type != 'mllama':
                vision_prompt_table = embeddings
                vision_prompt_vocab_size = np.array(
                    [[vision_prompt_table.shape[1]]])
                # Concatenate the prompt tables if there are multiple images in single request
                if num_images > 1:
                    prompt_table = vision_prompt_table.view(
                        batch_size, -1, vision_prompt_table.shape[-1])
                    prompt_vocab_size = np.repeat(vision_prompt_vocab_size,
                                                  batch_size,
                                                  axis=0)
                else:
                    # Use the single prompt table directly
                    vision_prompt_vocab_size = np.repeat(
                        vision_prompt_vocab_size, batch_size, axis=0)
                    prompt_table = vision_prompt_table
                    prompt_vocab_size = vision_prompt_vocab_size

                prompt_embedding_table_tensor = pb_utils.Tensor.from_dlpack(
                    'OUT_PROMPT_EMBEDDING_TABLE', to_dlpack(prompt_table))

                prompt_vocab_size_tensor = pb_utils.Tensor(
                    'OUT_PROMPT_VOCAB_SIZE',
                    prompt_vocab_size.astype(np.int32))

                output_tensors = [
                    prompt_embedding_table_tensor, prompt_vocab_size_tensor
                ]
            else:
                max_tokens = pb_utils.get_input_tensor_by_name(
                    request, 'max_tokens')
                # max_tokens is needed to prepare the cross_attention_mask
                max_tokens = 0 if max_tokens is None else max_tokens.as_numpy(
                )[0, 0]

                # reshape encoder output
                # [bs, num_images, num_tiles, num_patches, hidden_size] to [bs, encoder_length, hidden_size]
                encoder_input_features = embeddings
                output_shape = encoder_input_features.shape
                encoder_input_features = encoder_input_features.reshape(
                    output_shape[0],
                    output_shape[1] * output_shape[2] * output_shape[3],
                    output_shape[4])
                logger.debug(
                    f"encoder_input_features shape: {encoder_input_features.shape}"
                )

                # prepare encoder output lengths
                # shape [bs], value [encoder_length]

                encoder_output_lengths = torch.tensor(
                    [[output_shape[1] * output_shape[2] * output_shape[3]]],
                    dtype=torch.int32)
                logger.debug(
                    f"encoder_output_lengths: {encoder_output_lengths}")
                skip_cross_attn_blocks = torch.ones([output_shape[0], 1],
                                                    dtype=torch.bool,
                                                    device='cpu')
                logger.debug(
                    f"skip_cross_attn_blocks: {skip_cross_attn_blocks}")

                # prepare cross_attention_mask
                # [bs, seq_len, num_tiles] to [bs, seq_len+max_new_tokens, encoder_length]
                cross_attention_mask = pb_utils.get_input_tensor_by_name(
                    request, "cross_attention_mask")
                if cross_attention_mask != None:
                    cross_attention_mask = from_dlpack(
                        pb_utils.get_input_tensor_by_name(
                            request, "cross_attention_mask").to_dlpack())
                    cross_attention_mask = cross_attention_mask.repeat_interleave(
                        output_shape[3], dim=3)
                    cross_attention_mask = cross_attention_mask.to(
                        encoder_input_features.device).to(torch.bool).reshape([
                            output_shape[0], -1,
                            encoder_input_features.shape[1]
                        ])
                    tmp_mask = [cross_attention_mask] + [
                        cross_attention_mask[:, -1:, :]
                        for _ in range(max_tokens)
                    ]
                    cross_attention_mask = torch.concat(tmp_mask, dim=1)
                    logger.debug(
                        f"cross attention mask shape: {cross_attention_mask.shape}"
                    )

                output_tensors.append(
                    pb_utils.Tensor.from_dlpack(
                        'ENCODER_INPUT_FEATURES',
                        to_dlpack(encoder_input_features)))
                output_tensors.append(
                    pb_utils.Tensor.from_dlpack(
                        'ENCODER_OUTPUT_LENGTHS',
                        to_dlpack(encoder_output_lengths)))
                if cross_attention_mask is not None:
                    output_tensors.append(
                        pb_utils.Tensor.from_dlpack(
                            'CROSS_ATTENTION_MASK',
                            to_dlpack(cross_attention_mask)))
                output_tensors.append(
                    pb_utils.Tensor.from_dlpack(
                        'SKIP_CROSS_ATTN_BLOCKS',
                        to_dlpack(skip_cross_attn_blocks)))

            inference_response = pb_utils.InferenceResponse(
                output_tensors=output_tensors)
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
