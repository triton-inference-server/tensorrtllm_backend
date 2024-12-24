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

import math
import sys
from unittest.mock import MagicMock

import pytest
import torch

sys.modules["triton_python_backend_utils"] = MagicMock()
# Use PYTHONPATH=../multimodal/multimodal_encoders
from multimodal_utils import LlavaOnevisionUtils


@pytest.fixture
def LlavaOvUtiils():
    # MockConfig
    class VisionConfig:

        def __init__(self, image_size, patch_size):
            self.image_size = image_size
            self.patch_size = patch_size

    class Config:

        def __init__(self, vision_aspect_ratio, image_grid_pinpoints,
                     hidden_size, vision_config):
            self.vision_aspect_ratio = vision_aspect_ratio
            self.image_grid_pinpoints = image_grid_pinpoints
            self.vision_config = vision_config
            self.hidden_size = hidden_size

    grid_pinpoints = [[1152, 1536], [1152, 1152]]
    vision_config = VisionConfig(384, 14)
    config = Config("anyres_max_9", grid_pinpoints, 3584, vision_config)
    newline = torch.ones((3584), dtype=torch.float16)

    return LlavaOnevisionUtils(config, newline)


# Test for LlavaOnevisionUtils.postprocess_video()
@pytest.mark.parametrize("image_features, batch_size, frames",
                         [(torch.ones(
                             (8, 729, 3584), dtype=torch.float16), 1, 8),
                          (torch.ones(
                              (20, 729, 3584), dtype=torch.float16), 2, 10)])
def test_llava_onevision_utils_video(LlavaOvUtiils, image_features, batch_size,
                                     frames):
    output = LlavaOvUtiils.postprocess_video(image_features, batch_size,
                                             frames)

    hw = LlavaOvUtiils.config.vision_config.image_size // LlavaOvUtiils.config.vision_config.patch_size
    size = hw * hw
    dim = math.ceil(math.sqrt(size) / 2)
    expected_dim = frames * dim * dim + 1
    assert output.dtype == image_features.dtype
    assert output.dim() == 3
    assert output.shape[0] == batch_size
    assert output.shape[1] == expected_dim
    assert output.shape[2] == LlavaOvUtiils.config.hidden_size


# Test for LlavaOnevisionUtils.postprocess_image()
@pytest.mark.parametrize(
    "image_features, image_sizes, image_num_patches, expected_dim",
    [(torch.ones(
        (13, 729, 3584), dtype=torch.float16), [[874, 1192]], [13], 7284),
     (torch.ones(
         (20, 729, 3584), dtype=torch.float16), [[899, 1024], [899, 1024]
                                                 ], [10, 10], 6551)])
def test_llava_onevision_utils_image(LlavaOvUtiils, image_features,
                                     image_sizes, image_num_patches,
                                     expected_dim):
    output = LlavaOvUtiils.postprocess_image(image_features, image_sizes,
                                             image_num_patches)

    assert output.dtype == image_features.dtype
    assert output.dim() == 3
    assert output.shape[0] == len(image_sizes)
    assert output.shape[1] == expected_dim
    assert output.shape[2] == LlavaOvUtiils.config.hidden_size
