## End to end workflow to run llama 7b

0. Make sure that you have initialized the TRT-LLM submodule:

```bash
git lfs install
git submodule update --init --recursive
```

1. (Optional) Download the LLaMa model from HuggingFace:

```bash
huggingface-cli login

huggingface-cli download meta-llama/Llama-2-7b-hf
```

> **NOTE**
>
> Make sure that you have access to https://huggingface.co/meta-llama/Llama-2-7b-hf.

2. Start the Triton Server Docker container:

```bash
# Replace <yy.mm> with the version of Triton you want to use.
# The command below assumes the the current directory is the
# TRT-LLM backend root git repository.

docker run --rm -ti -v `pwd`:/mnt -w /mnt -v ~/.cache/huggingface:~/.cache/huggingface --gpus all nvcr.io/nvidia/tritonserver:\<yy.mm\>-trtllm-python-py3 bash
```

3. Build the engine:
```bash
# Replace 'HF_LLAMA_MODE' with another path if you didn't download the model from step 1
# or you're not using HuggingFace.
export HF_LLAMA_MODEL=`python3 -c "from pathlib import Path; from huggingface_hub import hf_hub_download; print(Path(hf_hub_download('meta-llama/Llama-2-7b-hf', filename='config.json')).parent)"`
export UNIFIED_CKPT_PATH=/tmp/ckpt/llama/7b/
export ENGINE_PATH=/tmp/engines/llama/7b/
python tensorrt_llm/examples/models/core/llama/convert_checkpoint.py --model_dir ${HF_LLAMA_MODEL} \
                             --output_dir ${UNIFIED_CKPT_PATH} \
                             --dtype float16

trtllm-build --checkpoint_dir ${UNIFIED_CKPT_PATH} \
             --remove_input_padding enable \
             --gpt_attention_plugin float16 \
             --context_fmha enable \
             --gemm_plugin float16 \
             --output_dir ${ENGINE_PATH} \
             --paged_kv_cache enable \
             --max_batch_size 64
```

* Prepare configs

```bash
cp tensorrt_llm/triton_backend/ci/all_models/inflight_batcher_llm/ llama_ifb -r

python3 tensorrt_llm/triton_backend/tools/fill_template.py -i llama_ifb/preprocessing/config.pbtxt tokenizer_dir:${HF_LLAMA_MODEL},triton_max_batch_size:64,preprocessing_instance_count:1
python3 tensorrt_llm/triton_backend/tools/fill_template.py -i llama_ifb/postprocessing/config.pbtxt tokenizer_dir:${HF_LLAMA_MODEL},triton_max_batch_size:64,postprocessing_instance_count:1
python3 tensorrt_llm/triton_backend/tools/fill_template.py -i llama_ifb/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:64,decoupled_mode:False,bls_instance_count:1,accumulate_tokens:False,logits_datatype:TYPE_FP32,prompt_embedding_table_data_type:TYPE_FP16
python3 tensorrt_llm/triton_backend/tools/fill_template.py -i llama_ifb/ensemble/config.pbtxt triton_max_batch_size:64,logits_datatype:TYPE_FP32
python3 tensorrt_llm/triton_backend/tools/fill_template.py -i llama_ifb/tensorrt_llm/config.pbtxt triton_backend:tensorrtllm,triton_max_batch_size:64,decoupled_mode:False,max_beam_width:1,engine_dir:${ENGINE_PATH},max_tokens_in_paged_kv_cache:2560,max_attention_window_size:2560,kv_cache_free_gpu_mem_fraction:0.5,exclude_input_in_output:True,enable_kv_cache_reuse:False,batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:0,encoder_input_features_data_type:TYPE_FP16,logits_datatype:TYPE_FP32,prompt_embedding_table_data_type:TYPE_FP16
```

* Launch server

```bash
pip install SentencePiece
python3 tensorrt_llm/triton_backend/scripts/launch_triton_server.py --world_size 1 --model_repo=llama_ifb/
```

this setting requires about 25GB

```bash
nvidia-smi

Wed Nov 29 08:51:30 2023
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA H100 PCIe               On  | 00000000:41:00.0 Off |                    0 |
| N/A   40C    P0              79W / 350W |  25169MiB / 81559MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
+---------------------------------------------------------------------------------------+
```

* Send request

```bash
curl -X POST localhost:8000/v2/models/ensemble/generate -d '{"text_input": "What is machine learning?", "max_tokens": 20, "bad_words": "", "stop_words": "", "pad_id": 2, "end_id": 2}'

{"cum_log_probs":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":"\nMachine learning is a subset of artificial intelligence (AI) that uses algorithms to learn from data and"}
```

* Send request with bad_words and stop_words

```bash
curl -X POST localhost:8000/v2/models/ensemble/generate -d '{"text_input": "What is machine learning?", "max_tokens": 20, "bad_words": ["intelligence", "model"], "stop_words": ["focuses", "learn"], "pad_id": 2, "end_id": 2}'

{"cum_log_probs":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":"\nMachine learning is a subset of artificial Intelligence (AI) that allows computers to learn"}
```

* Send request by `inflight_batcher_llm_client.py`

```bash
python3 tensorrt_llm/triton_backend/tools/inflight_batcher_llm/inflight_batcher_llm_client.py --request-output-len 200 --tokenizer-dir ${HF_LLAMA_MODEL}

=========
[[1, 19298, 297, 6641, 29899, 23027, 3444, 29892, 1105, 7598, 16370, 408, 263]]
Got completed request
Input: Born in north-east France, Soyer trained as a
Output beam 0: 850. He was the first chef to be hired by the newly opened Delmonico’s restaurant, where he worked for 10 years. He then opened his own restaurant, which was a huge success.
Soyer was a prolific writer and his books include The Gastronomic Regenerator (1854), The Gastronomic Regenerator and Cookery for the People (1855), The Cuisine of To-day (1859), The Cuisine of To-morrow (1864), The Cuisine of the Future (1867), The Cuisine of the Future (1873), The Cuisine of the Future (1874), The Cuisine of the Future (1875), The Cuisine of the Future (1876), The
output_ids =  [14547, 297, 3681, 322, 4517, 1434, 8401, 304, 1570, 3088, 297, 29871, 29896, 29947, 29945, 29900, 29889, 940, 471, 278, 937, 14547, 304, 367, 298, 2859, 491, 278, 15141, 6496, 5556, 3712, 1417, 30010, 29879, 27144, 29892, 988, 540, 3796, 363, 29871, 29896, 29900, 2440, 29889, 940, 769, 6496, 670, 1914, 27144, 29892, 607, 471, 263, 12176, 2551, 29889, 13, 6295, 7598, 471, 263, 410, 29880, 928, 9227, 322, 670, 8277, 3160, 450, 402, 7614, 4917, 293, 2169, 759, 1061, 313, 29896, 29947, 29945, 29946, 511, 450, 402, 7614, 4917, 293, 2169, 759, 1061, 322, 17278, 708, 363, 278, 11647, 313, 29896, 29947, 29945, 29945, 511, 450, 315, 4664, 457, 310, 1763, 29899, 3250, 313, 29896, 29947, 29945, 29929, 511, 450, 315, 4664, 457, 310, 1763, 29899, 26122, 313, 29896, 29947, 29953, 29946, 511, 450, 315, 4664, 457, 310, 278, 16367, 313, 29896, 29947, 29953, 29955, 511, 450, 315, 4664, 457, 310, 278, 16367, 313, 29896, 29947, 29955, 29941, 511, 450, 315, 4664, 457, 310, 278, 16367, 313, 29896, 29947, 29955, 29946, 511, 450, 315, 4664, 457, 310, 278, 16367, 313, 29896, 29947, 29955, 29945, 511, 450, 315, 4664, 457, 310, 278, 16367, 313, 29896, 29947, 29955, 29953, 511, 450]
```

* Run test on dataset

```
python3 tensorrt_llm/triton_backend/tools/inflight_batcher_llm/end_to_end_test.py --dataset tensorrt_llm/triton_backend/ci/L0_backend_trtllm/simple_data.json --max-input-len 500

[INFO] Start testing on 13 prompts.
[INFO] Functionality test succeed.
[INFO] Warm up for benchmarking.
[INFO] Start benchmarking on 13 prompts.
[INFO] Total Latency: 962.179 ms
```



* Run with decoupled mode (streaming)

```bash
cp tensorrt_llm/triton_backend/ci/all_models/inflight_batcher_llm/ llama_ifb -r

python3 tensorrt_llm/triton_backend/tools/fill_template.py -i llama_ifb/preprocessing/config.pbtxt tokenizer_dir:${HF_LLAMA_MODEL},triton_max_batch_size:64,preprocessing_instance_count:1
python3 tensorrt_llm/triton_backend/tools/fill_template.py -i llama_ifb/postprocessing/config.pbtxt tokenizer_dir:${HF_LLAMA_MODEL},triton_max_batch_size:64,postprocessing_instance_count:1
python3 tensorrt_llm/triton_backend/tools/fill_template.py -i llama_ifb/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:64,decoupled_mode:True,bls_instance_count:1,accumulate_tokens:Truelogits_datatype:TYPE_FP32,prompt_embedding_table_data_type:TYPE_FP16
python3 tensorrt_llm/triton_backend/tools/fill_template.py -i llama_ifb/ensemble/config.pbtxt triton_max_batch_size:64,logits_datatype:TYPE_FP32
python3 tensorrt_llm/triton_backend/tools/fill_template.py -i llama_ifb/tensorrt_llm/config.pbtxt triton_backend:tensorrtllm,triton_max_batch_size:64,decoupled_mode:True,max_beam_width:1,engine_dir:${ENGINE_PATH},max_tokens_in_paged_kv_cache:2560,max_attention_window_size:2560,kv_cache_free_gpu_mem_fraction:0.5,exclude_input_in_output:True,enable_kv_cache_reuse:False,batching_strategy:inflight_batching,max_queue_delay_microseconds:0,encoder_input_features_data_type:TYPE_FP16,logits_datatype:TYPE_FP32,prompt_embedding_table_data_type:TYPE_FP16

pip install SentencePiece
python3 tensorrt_llm/triton_backend/scripts/launch_triton_server.py --world_size 1 --model_repo=llama_ifb/

python3 tensorrt_llm/triton_backend/tools/inflight_batcher_llm/inflight_batcher_llm_client.py --request-output-len 200 --tokenizer-dir ${HF_LLAMA_MODEL} --streaming
```

<details>
<summary> The result would be like
</summary>

```bash
=========
Input sequence:  [1, 19298, 297, 6641, 29899, 23027, 3444, 29892, 1105, 7598, 16370, 408, 263]
[14547]
[297]
[3681]
[322]
[4517]
[1434]
[8401]
[304]
[1570]
[3088]
[297]
[29871]
[29896]
[29947]
[29945]
[29900]
[29889]
[940]
[471]
[278]
[937]
[14547]
[304]
[367]
[298]
[2859]
[491]
[278]
[15141]
[6496]
[5556]
[3712]
[1417]
[30010]
[29879]
[27144]
[29892]
[988]
[540]
[3796]
[363]
[29871]
[29896]
[29900]
[2440]
[29889]
[940]
[769]
[6496]
[670]
[1914]
[27144]
[29892]
[607]
[471]
[263]
[12176]
[2551]
[29889]
[13]
[6295]
[7598]
[471]
[263]
[410]
[29880]
[928]
[9227]
[322]
[670]
[8277]
[3160]
[450]
[402]
[7614]
[4917]
[293]
[2169]
[759]
[1061]
[313]
[29896]
[29947]
[29945]
[29946]
[511]
[450]
[402]
[7614]
[4917]
[293]
[2169]
[759]
[1061]
[322]
[17278]
[708]
[363]
[278]
[11647]
[313]
[29896]
[29947]
[29945]
[29945]
[511]
[450]
[315]
[4664]
[457]
[310]
[1763]
[29899]
[3250]
[313]
[29896]
[29947]
[29945]
[29929]
[511]
[450]
[315]
[4664]
[457]
[310]
[1763]
[29899]
[26122]
[313]
[29896]
[29947]
[29953]
[29946]
[511]
[450]
[315]
[4664]
[457]
[310]
[278]
[16367]
[313]
[29896]
[29947]
[29953]
[29955]
[511]
[450]
[315]
[4664]
[457]
[310]
[278]
[16367]
[313]
[29896]
[29947]
[29955]
[29941]
[511]
[450]
[315]
[4664]
[457]
[310]
[278]
[16367]
[313]
[29896]
[29947]
[29955]
[29946]
[511]
[450]
[315]
[4664]
[457]
[310]
[278]
[16367]
[313]
[29896]
[29947]
[29955]
[29945]
[511]
[450]
[315]
[4664]
[457]
[310]
[278]
[16367]
[313]
[29896]
[29947]
[29955]
[29953]
[511]
[450]
Input: Born in north-east France, Soyer trained as a
Output beam 0: chef in Paris and London before moving to New York in 1850. He was the first chef to be hired by the newly opened Delmonico’s restaurant, where he worked for 10 years. He then opened his own restaurant, which was a huge success.
Soyer was a prolific writer and his books include The Gastronomic Regenerator (1854), The Gastronomic Regenerator and Cookery for the People (1855), The Cuisine of To-day (1859), The Cuisine of To-morrow (1864), The Cuisine of the Future (1867), The Cuisine of the Future (1873), The Cuisine of the Future (1874), The Cuisine of the Future (1875), The Cuisine of the Future (1876), The
Output sequence:  [1, 19298, 297, 6641, 29899, 23027, 3444, 29892, 1105, 7598, 16370, 408, 263, 14547, 297, 3681, 322, 4517, 1434, 8401, 304, 1570, 3088, 297, 29871, 29896, 29947, 29945, 29900, 29889, 940, 471, 278, 937, 14547, 304, 367, 298, 2859, 491, 278, 15141, 6496, 5556, 3712, 1417, 30010, 29879, 27144, 29892, 988, 540, 3796, 363, 29871, 29896, 29900, 2440, 29889, 940, 769, 6496, 670, 1914, 27144, 29892, 607, 471, 263, 12176, 2551, 29889, 13, 6295, 7598, 471, 263, 410, 29880, 928, 9227, 322, 670, 8277, 3160, 450, 402, 7614, 4917, 293, 2169, 759, 1061, 313, 29896, 29947, 29945, 29946, 511, 450, 402, 7614, 4917, 293, 2169, 759, 1061, 322, 17278, 708, 363, 278, 11647, 313, 29896, 29947, 29945, 29945, 511, 450, 315, 4664, 457, 310, 1763, 29899, 3250, 313, 29896, 29947, 29945, 29929, 511, 450, 315, 4664, 457, 310, 1763, 29899, 26122, 313, 29896, 29947, 29953, 29946, 511, 450, 315, 4664, 457, 310, 278, 16367, 313, 29896, 29947, 29953, 29955, 511, 450, 315, 4664, 457, 310, 278, 16367, 313, 29896, 29947, 29955, 29941, 511, 450, 315, 4664, 457, 310, 278, 16367, 313, 29896, 29947, 29955, 29946, 511, 450, 315, 4664, 457, 310, 278, 16367, 313, 29896, 29947, 29955, 29945, 511, 450, 315, 4664, 457, 310, 278, 16367, 313, 29896, 29947, 29955, 29953, 511, 450]
```

</details>


* Run several requests at the same time

```bash
echo '{"text_input": "What is machine learning?", "max_tokens": 20, "bad_words": "", "stop_words": "", "pad_id": 2, "end_id": 2}' > tmp.txt
printf '%s\n' {1..20} | xargs -I % -P 20 curl -X POST localhost:8000/v2/models/ensemble/generate -d @tmp.txt
```
