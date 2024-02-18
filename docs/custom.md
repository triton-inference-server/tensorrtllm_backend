## End to end workflow to run sp model

* Build engine

<!-- TODO add more steps -->
```bash
export TOKENIZER_DIR=/home/scratch.bhsueh_sw_1/workspace/TensorRT-LLM/deepmind_2/models/taurus_nv/checkpoints/tmp_vocab.model

```

* Prepare configs

Note that we use `tokenizer_type=sp` (sentencepiece) tokenizer.

```bash
cp all_models/inflight_batcher_llm/ sp_model -r

python3 tools/fill_template.py -i sp_model/preprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_DIR},tokenizer_type:sp,triton_max_batch_size:64,preprocessing_instance_count:1
python3 tools/fill_template.py -i sp_model/postprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_DIR},tokenizer_type:sp,triton_max_batch_size:64,postprocessing_instance_count:1
python3 tools/fill_template.py -i sp_model/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:64,decoupled_mode:False,bls_instance_count:1,accumulate_tokens:False
python3 tools/fill_template.py -i sp_model/ensemble/config.pbtxt triton_max_batch_size:64
python3 tools/fill_template.py -i sp_model/tensorrt_llm/config.pbtxt triton_max_batch_size:64,decoupled_mode:False,max_beam_width:1,engine_dir:/tmp/taurus/2B/fp8/1-gpu/,max_tokens_in_paged_kv_cache:2560,max_attention_window_size:2560,kv_cache_free_gpu_mem_fraction:0.5,exclude_input_in_output:True,enable_kv_cache_reuse:False,batching_strategy:inflight_batching,max_queue_delay_microseconds:600,batch_scheduler_policy:guaranteed_no_evict,enable_trt_overlap:False

```

* Launch server

```bash
python3 scripts/launch_triton_server.py --world_size 1 --model_repo=sp_model/
```


* Send request

```bash
curl -X POST localhost:8000/v2/models/ensemble/generate -d '{"text_input": "What is machine learning?", "max_tokens": 20, "bad_words": "", "stop_words": ""}'

{"context_logits":0.0,"cum_log_probs":0.0,"generation_logits":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":"Machine learning (ML) is a type of artificial intelligence (AI) that enables computers to learn"}
```

* Send request with bad_words and stop_words

```bash
curl -X POST localhost:8000/v2/models/ensemble/generate -d '{"text_input": "What is machine learning?", "max_tokens": 20, "bad_words": [" intelligence", " enables"], "stop_words": [" computers", "learn"]}'

{"context_logits":0.0,"cum_log_probs":0.0,"generation_logits":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":"Machine learning (ML) is a type of artificial Intelligence (AI) that allows computers"}
```

The words ` intelligence` and ` enables` are replaced by ` Intelligence` and ` allows`, and the generation stops when generating ` computers`.
