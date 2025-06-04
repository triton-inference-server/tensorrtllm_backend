# Model Configuration

## Model Parameters

The following tables show the parameters in the `config.pbtxt` of the models in
[all_models/inflight_batcher_llm](https://github.com/NVIDIA/TensorRT-LLM/blob/main/triton_backend/all_models/inflight_batcher_llm).
that can be modified before deployment. For optimal performance or custom
parameters, please refer to
[perf_best_practices](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/performance/perf-best-practices.md).

The names of the parameters listed below are the values in the `config.pbtxt`
that can be modified using the
[`fill_template.py`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/triton_backend/tools/fill_template.py) script.

**NOTE** For fields that have comma as the value (e.g. `gpu_device_ids`,
`participant_ids`), you need to escape the comma with
a backslash. For example, if you want to set `gpu_device_ids` to `0,1` you need
to run `python3 fill_template.py -i config.pbtxt "gpu_device_ids:0\,1".`

The mandatory parameters must be set for the model to run. The optional
parameters are not required but can be set to customize the model.

### ensemble model

See
[here](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/architecture.md#ensemble-models)
to learn more about ensemble models.

*Mandatory parameters*

| Name | Description |
| :----------------------: | :-----------------------------: |
| `triton_max_batch_size` | The maximum batch size that the Triton model instance will run with. Note that for the `tensorrt_llm` model, the actual runtime batch size can be larger than `triton_max_batch_size`. The runtime batch size will be determined by the TRT-LLM scheduler based on a number of parameters such as number of available requests in the queue, and the engine build `trtllm-build` parameters (such `max_num_tokens` and `max_batch_size`). |
| `logits_datatype` | The data type for context and generation logits. |

### preprocessing model

*Mandatory parameters*

| Name | Description |
| :----------------------: | :-----------------------------: |
| `triton_max_batch_size` | The maximum batch size that Triton should use with the model. |
| `tokenizer_dir` | The path to the tokenizer for the model. |
| `preprocessing_instance_count` | The number of instances of the model to run. |
| `max_queue_delay_microseconds` | The maximum queue delay in microseconds. Setting this parameter to a value greater than 0 can improve the chances that two requests arriving within `max_queue_delay_microseconds` will be scheduled in the same TRT-LLM iteration. |
| `max_queue_size` | The maximum number of requests allowed in the TRT-LLM queue before rejecting new requests. |

*Optional parameters*

| Name | Description |
| :----------------------: | :-----------------------------: |
| `add_special_tokens` | The `add_special_tokens` flag used by [HF tokenizers](https://huggingface.co/transformers/v2.11.0/main_classes/tokenizer.html#transformers.PreTrainedTokenizer.add_special_tokens). |
| `multimodal_model_path` | The vision engine path used in multimodal workflow. |
| `engine_dir` | The path to the engine for the model. This parameter is only needed for *multimodal processing* to extract the `vocab_size` from the engine_dir's config.json for `fake_prompt_id` mappings. |


### multimodal_encoders model

*Mandatory parameters*

| Name | Description |
| :----------------------: | :-----------------------------: |
| `triton_max_batch_size` | The maximum batch size that Triton should use with the model. |
| `max_queue_delay_microseconds` | The maximum queue delay in microseconds. Setting this parameter to a value greater than 0 can improve the chances that two requests arriving within `max_queue_delay_microseconds` will be scheduled in the same TRT-LLM iteration. |
| `max_queue_size` | The maximum number of requests allowed in the TRT-LLM queue before rejecting new requests. |
| `multimodal_model_path` | The vision engine path used in multimodal workflow. |
| `hf_model_path` | The Huggingface model path used for `llava_onevision` and `mllama` models. |


### postprocessing model

*Mandatory parameters*

| Name | Description |
| :----------------------: | :-----------------------------: |
| `triton_max_batch_size` | The maximum batch size that Triton should use with the model. |
| `tokenizer_dir` | The path to the tokenizer for the model. |
| `postprocessing_instance_count` | The number of instances of the model to run. |

*Optional parameters*

| Name | Description |
| :----------------------: | :-----------------------------: |
| `skip_special_tokens` | The `skip_special_tokens` flag used by [HF detokenizers](https://huggingface.co/transformers/v2.11.0/main_classes/tokenizer.html#transformers.PreTrainedTokenizer.decode). |

### tensorrt_llm model

The majority of the `tensorrt_llm` model parameters and input/output tensors
can be mapped to parameters in the TRT-LLM C++ runtime API defined in
[`executor.h`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/include/tensorrt_llm/executor/executor.h).
Please refer to the Doxygen comments in `executor.h` for a more detailed
description of the parameters below.

*Mandatory parameters*

| Name | Description |
| :----------------------: | :-----------------------------: |
| `triton_backend` | The backend to use for the model. Set to `tensorrtllm` to utilize the C++ TRT-LLM backend implementation. Set to `python` to utlize the TRT-LLM Python runtime. |
| `triton_max_batch_size` | The maximum batch size that the Triton model instance will run with. Note that for the `tensorrt_llm` model, the actual runtime batch size can be larger than `triton_max_batch_size`. The runtime batch size will be determined by the TRT-LLM scheduler based on a number of parameters such as number of available requests in the queue, and the engine build `trtllm-build` parameters (such `max_num_tokens` and `max_batch_size`). |
| `decoupled_mode` | Whether to use decoupled mode. Must be set to `true` for requests setting the `stream` tensor to `true`. |
| `max_queue_delay_microseconds` | The maximum queue delay in microseconds. Setting this parameter to a value greater than 0 can improve the chances that two requests arriving within `max_queue_delay_microseconds` will be scheduled in the same TRT-LLM iteration. |
| `max_queue_size` | The maximum number of requests allowed in the TRT-LLM queue before rejecting new requests. |
| `engine_dir` | The path to the engine for the model. |
| `batching_strategy` | The batching strategy to use. Set to `inflight_fused_batching` when enabling in-flight batching support. To disable in-flight batching, set to `V1` |
| `encoder_input_features_data_type` | The dtype for the input tensor `encoder_input_features`. For the mllama model, this must be `TYPE_BF16`. For other models like whisper, this is `TYPE_FP16`. |
| `logits_datatype` | The data type for context and generation logits. |

*Optional parameters*

- General

| Name | Description |
| :----------------------: | :-----------------------------: |
| `encoder_engine_dir` | When running encoder-decoder models, this is the path to the folder that contains the model configuration and engine for the encoder model. |
| `max_attention_window_size` | When using techniques like sliding window attention, the maximum number of tokens that are attended to generate one token. Defaults attends to all tokens in sequence. (default=max_sequence_length) |
| `sink_token_length` | Number of sink tokens to always keep in attention window. |
| `exclude_input_in_output` | Set to `true` to only return completion tokens in a response. Set to `false` to return the prompt tokens concatenated with the generated tokens. (default=`false`) |
| `cancellation_check_period_ms` | The time for cancellation check thread to sleep before doing the next check. It checks if any of the current active requests are cancelled through triton and prevent further execution of them. (default=100) |
| `stats_check_period_ms` | The time for the statistics reporting thread to sleep before doing the next check. (default=100) |
| `recv_poll_period_ms` | The time for the receiving thread in orchestrator mode to sleep before doing the next check. (default=0) |
| `iter_stats_max_iterations` | The maximum number of iterations for which to keep statistics. (default=ExecutorConfig::kDefaultIterStatsMaxIterations) |
| `request_stats_max_iterations` | The maximum number of iterations for which to keep per-request statistics. (default=executor::kDefaultRequestStatsMaxIterations) |
| `normalize_log_probs` | Controls if log probabilities should be normalized or not. Set to `false` to skip normalization of `output_log_probs`. (default=`true`) |
| `gpu_device_ids` | Comma-separated list of GPU IDs to use for this model. Use semicolons to separate multiple instances of the model. If not provided, the model will use all visible GPUs. (default=unspecified) |
| `participant_ids` | Comma-separated list of MPI ranks to use for this model. Mandatory when using orchestrator mode with -disable-spawn-process (default=unspecified) |
| `num_nodes` | Number of MPI nodes to use for this model. (default=1) |
| `gpu_weights_percent` | Set to a number between 0.0 and 1.0 to specify the percentage of weights that reside on GPU instead of CPU and streaming load during runtime. Values less than 1.0 are only supported for an engine built with `weight_streaming` on. (default=1.0) |

- KV cache

Note that the parameter `enable_trt_overlap` has been removed from the
config.pbtxt. This option allowed to overlap execution of two micro-batches to
hide CPU overhead. Optimization work has been done to reduce the CPU overhead
and it was found that the overlapping of micro-batches did not provide
additional benefits.

| Name | Description |
| :----------------------: | :-----------------------------: |
| `max_tokens_in_paged_kv_cache` | The maximum size of the KV cache in number of tokens. If unspecified, value is interpreted as 'infinite'. KV cache allocation is the min of max_tokens_in_paged_kv_cache and value derived from kv_cache_free_gpu_mem_fraction below. (default=unspecified) |
| `kv_cache_free_gpu_mem_fraction` | Set to a number between 0 and 1 to indicate the maximum fraction of GPU memory (after loading the model) that may be used for KV cache. (default=0.9) |
| `cross_kv_cache_fraction` | Set to a number between 0 and 1 to indicate the maximum fraction of KV cache that may be used for cross attention, and the rest will be used for self attention. Optional param and should be set for encoder-decoder models ONLY. (default=0.5) |
| `kv_cache_host_memory_bytes` |  Enable offloading to host memory for the given byte size. |
| `enable_kv_cache_reuse` | Set to `true` to reuse previously computed KV cache values (e.g. for system prompt) |

- LoRA cache

| Name | Description |
| :----------------------: | :-----------------------------: |
| `lora_cache_optimal_adapter_size` | Optimal adapter size used to size cache pages. Typically optimally sized adapters will fix exactly into 1 cache page. (default=8) |
| `lora_cache_max_adapter_size` | Used to set the minimum size of a cache page.  Pages must be at least large enough to fit a single module, single later adapter_size `maxAdapterSize` row of weights. (default=64) |
| `lora_cache_gpu_memory_fraction` | Fraction of GPU memory used for LoRA cache. Computed as a fraction of left over memory after engine load, and after KV cache is loaded. (default=0.05) |
| `lora_cache_host_memory_bytes` | Size of host LoRA cache in bytes. (default=1G) |
| `lora_prefetch_dir` | Folder to store the LoRA weights we hope to load during engine initialization. |

- Decoding mode

| Name | Description |
| :----------------------: | :-----------------------------: |
| `max_beam_width` | The beam width value of requests that will be sent to the executor. (default=1) |
| `decoding_mode` | Set to one of the following: `{top_k, top_p, top_k_top_p, beam_search, medusa, redrafter, lookahead, eagle}` to select the decoding mode. The `top_k` mode exclusively uses Top-K algorithm for sampling, The `top_p` mode uses exclusively Top-P algorithm for sampling. The top_k_top_p mode employs both Top-K and Top-P algorithms, depending on the runtime sampling params of the request. Note that the `top_k_top_p option` requires more memory and has a longer runtime than using `top_k` or `top_p` individually; therefore, it should be used only when necessary. `beam_search` uses beam search algorithm. If not specified, the default is to use `top_k_top_p` if `max_beam_width == 1`; otherwise, `beam_search` is used. When Medusa model is used, `medusa` decoding mode should be set. However, TensorRT-LLM detects loaded Medusa model and overwrites decoding mode to `medusa` with warning. Same applies to the ReDrafter, Lookahead and Eagle. |

- Optimization

| Name | Description |
| :----------------------: | :-----------------------------: |
| `enable_chunked_context` | Set to `true` to enable context chunking. (default=`false`) |
| `multi_block_mode` | Set to `false` to disable multi block mode. (default=`true`) |
| `enable_context_fmha_fp32_acc` | Set to `true` to enable FMHA runner FP32 accumulation. (default=`false`) |
| `cuda_graph_mode` | Set to `true` to enable cuda graph. (default=`false`) |
| `cuda_graph_cache_size` | Sets the size of the CUDA graph cache, in numbers of CUDA graphs. (default=0) |

- Scheduling

| Name | Description |
| :----------------------: | :-----------------------------: |
| `batch_scheduler_policy` | Set to `max_utilization` to greedily pack as many requests as possible in each current in-flight batching iteration. This maximizes the throughput but may result in overheads due to request pause/resume if KV cache limits are reached during execution. Set to `guaranteed_no_evict` to guarantee that a started request is never paused. (default=`guaranteed_no_evict`) |

- Medusa

| Name | Description |
| :----------------------: | :-----------------------------: |
| `medusa_choices` | To specify Medusa choices tree in the format of e.g. "{0, 0, 0}, {0, 1}". By default, `mc_sim_7b_63` choices are used. |

- Eagle

| Name | Description |
| :----------------------: | :-----------------------------: |
| `eagle_choices` | To specify default per-server Eagle choices tree in the format of e.g. "{0, 0, 0}, {0, 1}". By default, `mc_sim_7b_63` choices are used. |

- Guided decoding

| Name | Description |
| :----------------------: | :-----------------------------: |
| `guided_decoding_backend` | Set to `xgrammar` to activate guided decoder. |
| `tokenizer_dir` | The guided decoding of tensorrt_llm python backend requires tokenizer's information. |
| `xgrammar_tokenizer_info_path` | The guided decoding of tensorrt_llm C++ backend requires xgrammar's tokenizer's info in 'json' format. |

### tensorrt_llm_bls model

See
[here](https://github.com/triton-inference-server/python_backend#business-logic-scripting)
to learn more about BLS models.

*Mandatory parameters*

| Name | Description |
| :----------------------: | :-----------------------------: |
| `triton_max_batch_size` | The maximum batch size that the model can handle. |
| `decoupled_mode` | Whether to use decoupled mode. |
| `bls_instance_count` | The number of instances of the model to run. When using the BLS model instead of the ensemble, you should set the number of model instances to the maximum batch size supported by the TRT engine to allow concurrent request execution. |
| `logits_datatype` | The data type for context and generation logits. |

*Optional parameters*

- General

| Name | Description |
| :----------------------: | :-----------------------------: |
| `accumulate_tokens` | Used in the streaming mode to call the postprocessing model with all accumulated tokens, instead of only one token. This might be necessary for certain tokenizers. |

- Speculative decoding

The BLS model supports speculative decoding. Target and draft triton models are set with the parameters `tensorrt_llm_model_name` `tensorrt_llm_draft_model_name`. Speculative decodingis performed by setting `num_draft_tokens` in the request.  `use_draft_logits` may be set to use logits comparison speculative decoding. Note that `return_generation_logits` and `return_context_logits` are not supported when using speculative decoding. Also note that requests with batch size greater than 1 is not supported with speculative decoding right now.

| Name | Description |
| :----------------------: | :-----------------------------: |
| `tensorrt_llm_model_name` | The name of the TensorRT-LLM model to use. |
| `tensorrt_llm_draft_model_name` | The name of the TensorRT-LLM draft model to use. |

### Model Input and Output

Below is the lists of input and output tensors for the `tensorrt_llm` and
`tensorrt_llm_bls` models.

#### Common Inputs

| Name | Shape | Type | Description |
| :------------: | :---------------: | :-----------: | :--------: |
| `end_id` | [1] | `int32` | End token ID. If not specified, defaults to -1 |
| `pad_id` | [1] | `int32` | Padding token ID |
| `temperature` | [1] | `float32` | Sampling Config param: `temperature` |
| `repetition_penalty` | [1] | `float` | Sampling Config param: `repetitionPenalty` |
| `min_length` | [1] | `int32_t` | Sampling Config param: `minLength` |
| `presence_penalty` | [1] | `float` | Sampling Config param: `presencePenalty` |
| `frequency_penalty` | [1] | `float` | Sampling Config param: `frequencyPenalty` |
| `random_seed` | [1] | `uint64_t` | Sampling Config param: `randomSeed` |
| `return_log_probs` | [1] | `bool` | When `true`, include log probs in the output |
| `return_context_logits` | [1] | `bool` | When `true`, include context logits in the output |
| `return_generation_logits` | [1] | `bool` | When `true`, include generation logits in the output |
| `num_return_sequences` | [1] | `int32_t` | Number of generated sequences per request. (Default=1) |
| `beam_width` | [1] | `int32_t` | Beam width for this request; set to 1 for greedy sampling (Default=1) |
| `prompt_embedding_table` | [1] | `float16` (model data type) | P-tuning prompt embedding table |
| `prompt_vocab_size` | [1] | `int32` | P-tuning prompt vocab size |
| `return_perf_metrics` | [1] | `bool` | When `true`, include perf metrics in the output, such as kv cache reuse stats |
| `guided_decoding_guide_type` | [1] | `string` | Guided decoding param: `guide_type` |
| `guided_decoding_guide` | [1] | `string` | Guided decoding param: `guide` |

The following inputs for lora are for both `tensorrt_llm` and `tensorrt_llm_bls`
models. The inputs are passed through the `tensorrt_llm` model and the
`tensorrt_llm_bls` model will refer to the inputs from the `tensorrt_llm` model.

| Name | Shape | Type | Description |
| :------------: | :---------------: | :-----------: | :--------: |
| `lora_task_id` | [1] | `uint64` | The unique task ID for the given LoRA. To perform inference with a specific LoRA for the first time, `lora_task_id`, `lora_weights`, and `lora_config` must all be given. The LoRA will be cached, so that subsequent requests for the same task only require `lora_task_id`. If the cache is full, the oldest LoRA will be evicted to make space for new ones. An error is returned if `lora_task_id` is not cached |
| `lora_weights` | [ num_lora_modules_layers, D x Hi + Ho x D ] | `float` (model data type) | Weights for a LoRA adapter. See the config file for more details. |
| `lora_config` | [ num_lora_modules_layers, 3] | `int32t` | Module identifier. See the config file for more details. |

#### Common Outputs

Note: the timing metrics oputputs are represented as the number of nanoseconds since epoch.

| Name | Shape | Type | Description |
| :------------: | :---------------: | :-----------: | :--------: |
| `cum_log_probs` | [-1] | `float` | Cumulative probabilities for each output |
| `output_log_probs` | [beam_width, -1] | `float` | Log probabilities for each output |
| `context_logits` | [-1, vocab_size] | `float` | Context logits for input |
| `generation_logits` | [beam_width, seq_len, vocab_size] | `float` | Generation logits for each output |
| `batch_index` | [1] | `int32` | Batch index |
| `kv_cache_alloc_new_blocks` | [1] | `int32` | KV cache reuse metrics. Number of newly allocated blocks per request. Set the optional input `return_perf_metrics` to `true` to include `kv_cache_alloc_new_blocks` in the outputs. |
| `kv_cache_reused_blocks` | [1] | `int32` | KV cache reuse metrics. Number of reused blocks per request. Set the optional input `return_perf_metrics` to `true` to include `kv_cache_reused_blocks` in the outputs. |
| `kv_cache_alloc_total_blocks` | [1] | `int32` | KV cache reuse metrics. Number of total allocated blocks per request. Set the optional input `return_perf_metrics` to `true` to include `kv_cache_alloc_total_blocks` in the outputs. |
| `arrival_time_ns` | [1] | `float` | Time when the request was received by TRT-LLM. Set the optional input `return_perf_metrics` to `true` to include `arrival_time_ns` in the outputs. |
| `first_scheduled_time_ns` | [1] | `float` | Time when the request was first scheduled. Set the optional input `return_perf_metrics` to `true` to include `first_scheduled_time_ns` in the outputs. |
| `first_token_time_ns` | [1] | `float` | Time when the first token was generated. Set the optional input `return_perf_metrics` to `true` to include `first_token_time_ns` in the outputs. |
| `last_token_time_ns` | [1] | `float` | Time when the last token was generated. Set the optional input `return_perf_metrics` to `true` to include `last_token_time_ns` in the outputs. |
| `acceptance_rate` | [1] | `float` | Acceptance rate of the speculative decoding model. Set the optional input `return_perf_metrics` to `true` to include `acceptance_rate` in the outputs. |
| `total_accepted_draft_tokens` | [1] | `int32` | Number of tokens accepted by the target model in speculative decoding. Set the optional input `return_perf_metrics` to `true` to include `total_accepted_draft_tokens` in the outputs. |
| `total_draft_tokens` | [1] | `int32` | Maximum number of draft tokens acceptable by the target model in speculative decoding. Set the optional input `return_perf_metrics` to `true` to include `total_draft_tokens` in the outputs. |

#### Unique Inputs for tensorrt_llm model

| Name | Shape | Type | Description |
| :------------: | :---------------: | :-----------: | :--------: |
| `input_ids` | [-1] | `int32` | Input token IDs |
| `input_lengths` | [1] | `int32` | Input lengths |
| `request_output_len` | [1] | `int32` | Requested output length |
| `draft_input_ids` | [-1] | `int32` | Draft input IDs |
| `decoder_input_ids` | [-1] | `int32` | Decoder input IDs |
| `decoder_input_lengths` | [1] | `int32` | Decoder input lengths |
| `draft_logits` | [-1, -1] | `float32` | Draft logits |
| `draft_acceptance_threshold` | [1] | `float32` | Draft acceptance threshold |
| `stop_words_list` | [2, -1] | `int32` | List of stop words |
| `bad_words_list` | [2, -1] | `int32` | List of bad words |
| `embedding_bias` | [-1] | `string` | Embedding bias words |
| `runtime_top_k` | [1] | `int32` | Top-k value for runtime top-k sampling |
| `runtime_top_p` | [1] | `float32` | Top-p value for runtime top-p sampling |
| `runtime_top_p_min` | [1] | `float32` | Minimum value for runtime top-p sampling |
| `runtime_top_p_decay` | [1] | `float32` | Decay value for runtime top-p sampling |
| `runtime_top_p_reset_ids` | [1] | `int32` | Reset IDs for runtime top-p sampling |
| `len_penalty` | [1] | `float32` | Controls how to penalize longer sequences in beam search (Default=0.f) |
| `early_stopping` | [1] | `bool` | Enable early stopping |
| `beam_search_diversity_rate` | [1] | `float32` | Beam search diversity rate |
| `stop` | [1] | `bool` | Stop flag |
| `streaming` | [1] | `bool` | Enable streaming |

#### Unique Outputs for tensorrt_llm model

| Name | Shape | Type | Description |
| :------------: | :---------------: | :-----------: | :--------: |
| `output_ids` | [-1, -1] | `int32` | Output token IDs |
| `sequence_length` | [-1] | `int32` | Sequence length |

#### Unique Inputs for tensorrt_llm_bls model

| Name | Shape | Type | Description |
| :------------: | :---------------: | :-----------: | :--------: |
| `text_input` | [-1] | `string` | Prompt text |
| `decoder_text_input` | [1] | `string` | Decoder input text |
| `image_input` | [3, 224, 224] | `float16` | Input image |
| `max_tokens` | [-1] | `int32` | Number of tokens to generate |
| `bad_words` | [2, num_bad_words] | `int32` | Bad words list |
| `stop_words` | [2, num_stop_words] | `int32` | Stop words list |
| `top_k` | [1] | `int32` | Sampling Config param: `topK` |
| `top_p` | [1] | `float32` | Sampling Config param: `topP` |
| `length_penalty` | [1] | `float32` | Sampling Config param: `lengthPenalty` |
| `stream` | [1] | `bool` | When `true`, stream out tokens as they are generated. When `false` return only when the full generation has completed (Default=`false`) |
|`embedding_bias_words` | [-1] | `string` | Embedding bias words |
| `embedding_bias_weights` | [-1] | `float32` | Embedding bias weights |
| `num_draft_tokens` | [1] | `int32` | Number of tokens to get from draft model during speculative decoding |
| `use_draft_logits` | [1] | `bool` | Use logit comparison during speculative decoding |

#### Unique Outputs for tensorrt_llm_bls model

| Name | Shape | Type | Description |
| :------------: | :---------------: | :-----------: | :--------: |
| `text_output` | [-1] | `string` | Text output |

## Some tips for model configuration

Below are some tips for configuring models for optimal performance. These
recommendations are based on our experiments and may not apply to all use cases.
For guidance on other parameters, please refer to the
[perf_best_practices](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/performance/perf-best-practices.md).

- **Setting the `instance_count` for models to better utilize inflight batching**

   The `instance_count` parameter in the config.pbtxt file specifies the number
   of instances of the model to run. Ideally, this should be set to match the
   maximum batch size supported by the TRT engine, as this allows for concurrent
   request execution and reduces performance bottlenecks. However, it will also
   consume more CPU memory resources. While the optimal value isn't something we
   can determine in advance, it generally shouldn't be set to a very small
   value, such as 1.
   For most use cases, we have found that setting `instance_count` to 5 works
   well across a variety of workloads in our experiments.

- **Adjusting `max_batch_size` and `max_num_tokens` to optimize inflight batching**

  `max_batch_size` and `max_num_tokens` are important parameters for optimizing
  inflight batching. You can modify `max_batch_size` in the model configuration
  file, while `max_num_tokens` is set during the conversion to a TRT-LLM engine
  using the `trtllm-build` command. Tuning these parameters is necessary for
  different scenarios, and experimentation is currently the best approach to
  finding optimal values. Generally, the total number of requests should be
  lower than `max_batch_size`, and the total tokens should be less than
  `max_num_tokens`.
