
## End to end workflow to run speculative decoding(using llama model)

* Build target model engine

```bash
export TARGET_HF_LLAMA_MODEL=llama-7b-hf-chat/
export TARGET_UNIFIED_CKPT_PATH=/tmp/ckpt/llama/7b/
export TARGET_ENGINE_PATH=/tmp/engines/llama/7b/
python convert_checkpoint.py --model_dir ${TARGET_HF_LLAMA_MODEL} \
                             --output_dir ${TARGET_UNIFIED_CKPT_PATH} \
                             --dtype float16

trtllm-build --checkpoint_dir ${TARGET_UNIFIED_CKPT_PATH} \
             --remove_input_padding enable \
             --gpt_attention_plugin float16 \
             --context_fmha enable \
             --gemm_plugin float16 \
             --output_dir ${TARGET_ENGINE_PATH} \
             --paged_kv_cache enable \
             --max_batch_size 64 \
             --max_draft_len 10 \
             --use_paged_context_fmha enable
```

* Build draft model engine


```bash
export DRAFT_HF_LLAMA_MODEL=llama-68m/
export DRAFT_UNIFIED_CKPT_PATH=/tmp/ckpt/llama/68m/
export DRAFT_ENGINE_PATH=/tmp/engines/llama/68m/
python convert_checkpoint.py --model_dir ${DRAFT_HF_LLAMA_MODEL} \
                             --output_dir ${DRAFT_UNIFIED_CKPT_PATH} \
                             --dtype float16

trtllm-build --checkpoint_dir ${DRAFT_UNIFIED_CKPT_PATH} \
             --remove_input_padding enable \
             --gpt_attention_plugin float16 \
             --context_fmha enable \
             --gemm_plugin float16 \
             --output_dir ${DRAFT_ENGINE_PATH} \
             --paged_kv_cache enable \
             --max_batch_size 64 \
             --use_paged_context_fmha enable
```

For llama-68m model, you can download from https://huggingface.co/JackFram/llama-68m.


* Prepare configs


```bash
cp all_models/spec_decode llama_sp -r

python3 tools/fill_template.py -i llama_sp/preprocessing/config.pbtxt tokenizer_dir:${DRAFT_HF_LLAMA_MODEL},triton_max_batch_size:64,preprocessing_instance_count:1
python3 tools/fill_template.py -i llama_sp/postprocessing/config.pbtxt tokenizer_dir:${TARGET_HF_LLAMA_MODEL},triton_max_batch_size:64,postprocessing_instance_count:1
python3 tools/fill_template.py -i llama_sp/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:64,decoupled_mode:False,bls_instance_count:1,accumulate_tokens:False
python3 tools/fill_template.py -i llama_sp/ensemble/config.pbtxt triton_max_batch_size:64
python3 tools/fill_template.py -i llama_sp/tensorrt_llm_target/config.pbtxt triton_max_batch_size:64,decoupled_mode:False,max_beam_width:1,engine_dir:${TARGET_ENGINE_PATH},max_tokens_in_paged_kv_cache:2560,max_attention_window_size:2560,kv_cache_free_gpu_mem_fraction:0.5,enable_kv_cache_reuse:True,batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:0
python3 tools/fill_template.py -i llama_sp/tensorrt_llm_draft/config.pbtxt triton_max_batch_size:64,decoupled_mode:False,max_beam_width:1,engine_dir:${DRAFT_ENGINE_PATH},max_tokens_in_paged_kv_cache:2560,max_attention_window_size:2560,kv_cache_free_gpu_mem_fraction:0.5,enable_kv_cache_reuse:True,batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:0
```

* Launch server

```bash
pip install SentencePiece
python3 scripts/launch_triton_server.py --world_size 1 --model_repo=llama_sp/
```

* Send request

```bash
curl -X POST localhost:8000/v2/models/tensorrt_llm_bls/generate -d '{"text_input": "What is machine learning?", "max_tokens": 20, "bad_words": "", "stop_words": "", "num_draft_tokens":5}'

{"context_logits":0.0,"cum_log_probs":0.0,"generation_logits":0.0,"model_name":"tensorrt_llm_bls","model_version":"1","output_log_probs":[0.0,0.0,0.0],"text_output":"What is machine learning?\n\nMachine learning is a subfield of artificial intelligence (AI) that involves the use of algorithms"}
```

* Send request by `e2e_grpc_speculative_decoding_client.py`

```bash
python3 inflight_batcher_llm/client/e2e_grpc_speculative_decoding_client.py --url-target localhost:8001 -p "What is machine learning?" --draft-tensorrt-llm-model-name "tensorrt_llm_draft" --target-tensorrt-llm-model-name "tensorrt_llm_target" --output-len 20

=========
Final text:
 What is machine learning?

Machine learning is a subfield of artificial intelligence (AI) that involves the use of algorithms
```