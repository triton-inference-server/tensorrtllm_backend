ENGINE_PATH=$1

if [ $ENGINE_PATH ]; then
  :
else
	echo "ENGINE_PATH ${ENGINE_PATH} IS NOT EXISTED"
    exit
fi

echo "
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

name: \"tekit\"
backend: \"python\"
max_batch_size: 1024

dynamic_batching {
   max_queue_delay_microseconds: 50000
}

input [
  {
    name: \"input_ids\"
    data_type: TYPE_INT32
    dims: [ -1 ]
  },
  {
    name: \"request_output_len\"
    data_type: TYPE_UINT32
    dims: [ -1 ]
  },
  {
    name: \"runtime_top_k\"
    data_type: TYPE_UINT32
    dims: [ 1 ]
    reshape: { shape: [ ] }
    optional: true
  },
  {
    name: \"runtime_top_p\"
    data_type: TYPE_FP32
    dims: [ 1 ]
    reshape: { shape: [ ] }
    optional: true
  },
  {
    name: \"beam_search_diversity_rate\"
    data_type: TYPE_FP32
    dims: [ 1 ]
    reshape: { shape: [ ] }
    optional: true
  },
  {
    name: \"temperature\"
    data_type: TYPE_FP32
    dims: [ 1 ]
    reshape: { shape: [ ] }
    optional: true
  },
  {
    name: \"len_penalty\"
    data_type: TYPE_FP32
    dims: [ 1 ]
    reshape: { shape: [ ] }
    optional: true
  },
  {
    name: \"repetition_penalty\"
    data_type: TYPE_FP32
    dims: [ 1 ]
    reshape: { shape: [ ] }
    optional: true
  },
  {
    name: \"random_seed\"
    data_type: TYPE_UINT64
    dims: [ 1 ]
    reshape: { shape: [ ] }
    optional: true
  },
  {
    name: \"beam_width\"
    data_type: TYPE_UINT32
    dims: [ 1 ]
    reshape: { shape: [ ] }
    optional: true
  }
]
output [
  {
    name: \"output_ids\"
    data_type: TYPE_UINT32
    dims: [ -1 ]
  }
]
instance_group [
  {
    count: 1
    kind : KIND_CPU
  }
]
parameters {
  key: \"engine_dir\"
  value: {
    string_value: \"${ENGINE_PATH}\"
  }
}
parameters: {
  key: \"FORCE_CPU_ONLY_INPUT_TENSORS\"
  value: {
    string_value: \"no\"
  }
}
" > config.pbtxt
