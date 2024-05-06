
## End to end workflow to run baichuan

* Build engine

```bash
export HF_BAICHUAN_MODEL=Baichuan-13B-Chat/
python build.py --model_dir ${HF_BAICHUAN_MODEL} \
                --dtype float16 \
                --remove_input_padding \
                --use_gpt_attention_plugin float16 \
                --enable_context_fmha \
                --use_gemm_plugin float16 \
                --output_dir /tmp/baichuan/13B/trt_engines/fp16/1-gpu/ \
                --paged_kv_cache \
                --max_batch_size 64

[11/29/2023-08:20:34] [TRT] [I] Total Host Persistent Memory: 77008
[11/29/2023-08:20:34] [TRT] [I] Total Device Persistent Memory: 0
[11/29/2023-08:20:34] [TRT] [I] Total Scratch Memory: 1342439424
[11/29/2023-08:20:34] [TRT] [I] [BlockAssignment] Started assigning block shifts. This will take 690 steps to complete.
[11/29/2023-08:20:34] [TRT] [I] [BlockAssignment] Algorithm ShiftNTopDown took 25.5938ms to assign 11 blocks to 690 nodes requiring 6308236288 bytes.
[11/29/2023-08:20:34] [TRT] [I] Total Activation Memory: 6308236288
[11/29/2023-08:20:35] [TRT] [I] Total Weights Memory: 26529804072
[11/29/2023-08:20:35] [TRT] [I] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +0, GPU +64, now: CPU 56027, GPU 28529 (MiB)
[11/29/2023-08:20:35] [TRT] [I] [MemUsageChange] Init cuDNN: CPU +0, GPU +72, now: CPU 56027, GPU 28601 (MiB)
[11/29/2023-08:20:35] [TRT] [I] [MemUsageStats] Peak memory usage of TRT CPU/GPU memory allocators: CPU 1250 MiB, GPU 41088 MiB
[11/29/2023-08:20:35] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in building engine: CPU +0, GPU +25301, now: CPU 0, GPU 25301 (MiB)
[11/29/2023-08:20:44] [TRT] [I] [MemUsageStats] Peak memory usage during Engine building and serialization: CPU: 81260 MiB
[11/29/2023-08:20:44] [TRT-LLM] [I] Total time of building baichuan_float16_tp1_rank0.engine: 00:00:37
[11/29/2023-08:20:44] [TRT-LLM] [I] Config saved to /tmp/baichuan/13B/trt_engines/fp16/1-gpu/config.json.
[11/29/2023-08:20:45] [TRT-LLM] [I] Serializing engine to /tmp/baichuan/13B/trt_engines/fp16/1-gpu/baichuan_float16_tp1_rank0.engine...
[11/29/2023-08:21:35] [TRT-LLM] [I] Engine serialized. Total time: 00:00:49
[11/29/2023-08:21:36] [TRT-LLM] [I] Timing cache serialized to /tmp/baichuan/13B/trt_engines/fp16/1-gpu/model.cache
[11/29/2023-08:21:36] [TRT-LLM] [I] Total time of building all 1 engines: 00:05:00
```

* Prepare configs

```bash
cp all_models/inflight_batcher_llm/ baichuan_ifb -r

python3 tools/fill_template.py -i baichuan_ifb/preprocessing/config.pbtxt tokenizer_dir:${HF_BAICHUAN_MODEL},triton_max_batch_size:64,preprocessing_instance_count:1
python3 tools/fill_template.py -i baichuan_ifb/postprocessing/config.pbtxt tokenizer_dir:${HF_BAICHUAN_MODEL},triton_max_batch_size:64,postprocessing_instance_count:1
python3 tools/fill_template.py -i baichuan_ifb/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:64,decoupled_mode:False,bls_instance_count:1,accumulate_tokens:False
python3 tools/fill_template.py -i baichuan_ifb/ensemble/config.pbtxt triton_max_batch_size:64
python3 tools/fill_template.py -i baichuan_ifb/tensorrt_llm/config.pbtxt triton_backend:tensorrtllm,triton_max_batch_size:64,decoupled_mode:False,max_beam_width:1,engine_dir:/tmp/baichuan/13B/trt_engines/fp16/1-gpu/,max_tokens_in_paged_kv_cache:2560,max_attention_window_size:2560,kv_cache_free_gpu_mem_fraction:0.5,exclude_input_in_output:True,enable_kv_cache_reuse:False,batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:0
```

* Launch server

```bash
pip install SentencePiece
python3 scripts/launch_triton_server.py --world_size 1 --model_repo=baichuan_ifb/
```

this setting requires about 35GB

```bash
nvidia-smi

Wed Nov 29 08:33:50 2023
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA H100 PCIe               On  | 00000000:41:00.0 Off |                    0 |
| N/A   43C    P0              81W / 350W |  34743MiB / 81559MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
+---------------------------------------------------------------------------------------+
```

If you encounter error

```bash
I1129 08:28:33.267969 15088 model_lifecycle.cc:818] successfully loaded 'tensorrt_llm_bls'
I1129 08:28:33.928915 15088 pb_stub.cc:325] Failed to initialize Python stub: ValueError: Tokenizer class BaichuanTokenizer does not exist or is not currently imported.

At:
  /home/bhsueh/.local/lib/python3.10/site-packages/transformers/models/auto/tokenization_auto.py(748): from_pretrained
  /home/scratch.bhsueh_sw_1/workspace/TensorRT-LLM/tllm_backend_nvbug/baichuan_ifb/preprocessing/1/model.py(66): initialize

I1129 08:28:33.928991 15088 pb_stub.cc:325] Failed to initialize Python stub: ValueError: Tokenizer class BaichuanTokenizer does not exist or is not currently imported.

At:
  /home/bhsueh/.local/lib/python3.10/site-packages/transformers/models/auto/tokenization_auto.py(748): from_pretrained
  /home/scratch.bhsueh_sw_1/workspace/TensorRT-LLM/tllm_backend_nvbug/baichuan_ifb/postprocessing/1/model.py(65): initialize

E1129 08:28:34.285773 15088 backend_model.cc:634] ERROR: Failed to create instance: ValueError: Tokenizer class BaichuanTokenizer does not exist or is not currently imported.

At:
  /home/bhsueh/.local/lib/python3.10/site-packages/transformers/models/auto/tokenization_auto.py(748): from_pretrained
  /home/scratch.bhsueh_sw_1/workspace/TensorRT-LLM/tllm_backend_nvbug/baichuan_ifb/postprocessing/1/model.py(65): initialize

E1129 08:28:34.285879 15088 model_lifecycle.cc:621] failed to load 'postprocessing' version 1: Internal: ValueError: Tokenizer class BaichuanTokenizer does not exist or is not currently imported.

At:
  /home/bhsueh/.local/lib/python3.10/site-packages/transformers/models/auto/tokenization_auto.py(748): from_pretrained
  /home/scratch.bhsueh_sw_1/workspace/TensorRT-LLM/tllm_backend_nvbug/baichuan_ifb/postprocessing/1/model.py(65): initialize

I1129 08:28:34.285894 15088 model_lifecycle.cc:756] failed to load 'postprocessing'
E1129 08:28:34.304925 15088 backend_model.cc:634] ERROR: Failed to create instance: ValueError: Tokenizer class BaichuanTokenizer does not exist or is not currently imported.

At:
  /home/bhsueh/.local/lib/python3.10/site-packages/transformers/models/auto/tokenization_auto.py(748): from_pretrained
  /home/scratch.bhsueh_sw_1/workspace/TensorRT-LLM/tllm_backend_nvbug/baichuan_ifb/preprocessing/1/model.py(66): initialize

E1129 08:28:34.305028 15088 model_lifecycle.cc:621] failed to load 'preprocessing' version 1: Internal: ValueError: Tokenizer class BaichuanTokenizer does not exist or is not currently imported.

At:
  /home/bhsueh/.local/lib/python3.10/site-packages/transformers/models/auto/tokenization_auto.py(748): from_pretrained
  /home/scratch.bhsueh_sw_1/workspace/TensorRT-LLM/tllm_backend_nvbug/baichuan_ifb/preprocessing/1/model.py(66): initialize

I1129 08:28:34.305052 15088 model_lifecycle.cc:756] failed to load 'preprocessing'
```

please add `trust_remote_code=True` in tokenizer of preprocessing and postprocessing. Considering the security, we don't add it by default.

* Send request

```bash
curl -X POST localhost:8000/v2/models/ensemble/generate -d '{"text_input": "What is machine learning?", "max_tokens": 20, "bad_words": "", "stop_words": "", "pad_id": 2, "end_id": 2}'

{"cum_log_probs":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":"\nMachine learning is a subset of artificial intelligence (AI) that focuses on the"}
```

* Send request with bad_words and stop_words

```bash
curl -X POST localhost:8000/v2/models/ensemble/generate -d '{"text_input": "What is machine learning?", "max_tokens": 20, "bad_words": ["intelligence","model"], "stop_words": ["focuses","learn"], "pad_id": 2, "end_id": 2}'

{"cum_log_probs":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":"\nMachine learning is a subset of artificial intelligent (AI) that focuses"}
```

* Send request by `inflight_batcher_llm_client.py` (Remember to add `trust_remote_code=True` in tokenizer of `inflight_batcher_llm_client.py`)

```bash
python3 inflight_batcher_llm/client/inflight_batcher_llm_client.py --request-output-len 200 --tokenizer-dir ${HF_BAICHUAN_MODEL}

=========
Input sequence:  [16448, 677, 5611, 31136, 21309, 4746, 31125, 694, 1033, 653, 8808, 754, 650]
Got completed request
Input: Born in north-east France, Soyer trained as a
Output beam 0: . He became the chef at the Reform Club, and later at the Vegetarian Restaurant, where he pioneered the use of vegetables in fine dining. He also wrote a number of books, including The London Art of Cookery (1858), The Modern Housekeeper (1861), and The Compleat Housekeeper (1862).
Soyer was a strong supporter of the British National Rifle Association, and was a member of the organisation's council. He was also a member of the Reform Club, the Athenaeum, and the Rifle Club. He died in London in 1904.
Soyer was born in the village of Montigny-lès-Cormeilles, in the department of Aisne, France. He was the son of a baker, and was educated in the
Output sequence:  [16814, 677, 5621, 1412, 4514, 678, 2835, 677, 31106, 53, 60, 57, 59, 79, 1057, 3142, 656, 16814, 772, 656, 15824, 4305, 31125, 680, 2384, 772, 656, 9592, 1161, 8480, 13550, 807, 31125, 1238, 742, 11135, 2521, 656, 1226, 679, 8431, 3392, 677, 4816, 8946, 79, 1057, 982, 4251, 650, 1697, 679, 3594, 31125, 1516, 776, 2835, 2409, 679, 7782, 1620, 762, 53, 60, 57, 60, 1098, 776, 8753, 2542, 17655, 762, 53, 60, 58, 53, 1098, 680, 776, 1127, 1596, 658, 2542, 17655, 762, 53, 60, 58, 54, 31145, 79, 5, 31131, 1033, 653, 796, 650, 2427, 23747, 679, 656, 3681, 2024, 751, 19422, 2790, 728, 31125, 680, 796, 650, 2736, 679, 656, 1625, 4859, 31155, 31114, 7284, 79, 1057, 796, 982, 650, 2736, 679, 656, 15824, 4305, 31125, 656, 1996, 1179, 4302, 784, 31125, 680, 656, 751, 19422, 4305, 79, 1057, 4357, 677, 2835, 677, 31106, 53, 61, 52, 56, 79, 5, 31131, 1033, 653, 796, 4204, 677, 656, 6730, 679, 5136, 942, 31124, 31136, 31115, 16987, 31136, 31133, 908, 31107, 22542, 31125, 677, 656, 1664, 2049, 679, 703, 667, 1024, 31125, 4746, 79, 1057, 796, 656, 3652, 679, 650, 675, 3034, 31125, 680, 796, 18735, 677, 656]
```

* Run test on dataset

```
python3 tools/inflight_batcher_llm/end_to_end_test.py --dataset ci/L0_backend_trtllm/simple_data.json --max-input-len 500

[INFO] Start testing on 13 prompts.
[INFO] Functionality test succeed.
[INFO] Warm up for benchmarking.
[INFO] Start benchmarking on 13 prompts.
[INFO] Total Latency: 1598.328 ms
```

* Run with decoupled mode (streaming)

```bash
cp all_models/inflight_batcher_llm/ baichuan_ifb -r

python3 tools/fill_template.py -i baichuan_ifb/preprocessing/config.pbtxt tokenizer_dir:${HF_BAICHUAN_MODEL},triton_max_batch_size:64,preprocessing_instance_count:1
python3 tools/fill_template.py -i baichuan_ifb/postprocessing/config.pbtxt tokenizer_dir:${HF_BAICHUAN_MODEL},triton_max_batch_size:64,postprocessing_instance_count:1
python3 tools/fill_template.py -i baichuan_ifb/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:64,decoupled_mode:True,bls_instance_count:1,accumulate_tokens:True
python3 tools/fill_template.py -i baichuan_ifb/ensemble/config.pbtxt triton_max_batch_size:64
python3 tools/fill_template.py -i baichuan_ifb/tensorrt_llm/config.pbtxt triton_backend:tensorrtllm,triton_max_batch_size:64,decoupled_mode:True,max_beam_width:1,engine_dir:/tmp/baichuan/13B/trt_engines/fp16/1-gpu/,max_tokens_in_paged_kv_cache:2560,max_attention_window_size:2560,kv_cache_free_gpu_mem_fraction:0.5,exclude_input_in_output:True,enable_kv_cache_reuse:False,batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:0

pip install SentencePiece
# please add `trust_remote_code=True` in tokenizer of preprocessing and postprocessing. Considering the security, we don't add it by default.
python3 scripts/launch_triton_server.py --world_size 1 --model_repo=baichuan_ifb/

python3 inflight_batcher_llm/client/inflight_batcher_llm_client.py --request-output-len 200 --tokenizer-dir ${HF_BAICHUAN_MODEL} --streaming
```

<details>
<summary> The result would be like
</summary>

```bash
=========
Input sequence:  [16448, 677, 5611, 31136, 21309, 4746, 31125, 694, 1033, 653, 8808, 754, 650]
[16814]
[677]
[5621]
[1412]
[4514]
[678]
[2835]
[677]
[31106]
[53]
[60]
[57]
[59]
[79]
[1057]
[3142]
[656]
[16814]
[772]
[656]
[15824]
[4305]
[31125]
[680]
[2384]
[772]
[656]
[9592]
[1161]
[8480]
[13550]
[807]
[31125]
[1238]
[742]
[11135]
[2521]
[656]
[1226]
[679]
[8431]
[3392]
[677]
[4816]
[8946]
[79]
[1057]
[982]
[4251]
[650]
[1697]
[679]
[3594]
[31125]
[1516]
[776]
[2835]
[2409]
[679]
[7782]
[1620]
[762]
[53]
[60]
[57]
[60]
[1098]
[776]
[8753]
[2542]
[17655]
[762]
[53]
[60]
[58]
[53]
[1098]
[680]
[776]
[1127]
[1596]
[658]
[2542]
[17655]
[762]
[53]
[60]
[58]
[54]
[31145]
[79]
[5]
[31131]
[1033]
[653]
[796]
[650]
[2427]
[23747]
[679]
[656]
[3681]
[2024]
[751]
[19422]
[2790]
[728]
[31125]
[680]
[796]
[650]
[2736]
[679]
[656]
[1625]
[4859]
[31155]
[31114]
[7284]
[79]
[1057]
[796]
[982]
[650]
[2736]
[679]
[656]
[15824]
[4305]
[31125]
[656]
[1996]
[1179]
[4302]
[784]
[31125]
[680]
[656]
[751]
[19422]
[4305]
[79]
[1057]
[4357]
[677]
[2835]
[677]
[31106]
[53]
[61]
[52]
[56]
[79]
[5]
[31131]
[1033]
[653]
[796]
[4204]
[677]
[656]
[6730]
[679]
[5136]
[942]
[31124]
[31136]
[31115]
[16987]
[31136]
[31133]
[908]
[31107]
[22542]
[31125]
[677]
[656]
[1664]
[2049]
[679]
[703]
[667]
[1024]
[31125]
[4746]
[79]
[1057]
[796]
[656]
[3652]
[679]
[650]
[675]
[3034]
[31125]
[680]
[796]
[18735]
[677]
[656]
Input: Born in north-east France, Soyer trained as a
Output beam 0: chef in Paris before moving to London in 1857. He became the chef at the Reform Club, and later at the Vegetarian Restaurant, where he pioneered the use of vegetables in fine dining. He also wrote a number of books, including The London Art of Cookery (1858), The Modern Housekeeper (1861), and The Compleat Housekeeper (1862).
Soyer was a strong supporter of the British National Rifle Association, and was a member of the organisation's council. He was also a member of the Reform Club, the Athenaeum, and the Rifle Club. He died in London in 1904.
Soyer was born in the village of Montigny-lès-Cormeilles, in the department of Aisne, France. He was the son of a baker, and was educated in the
Output sequence:  [16448, 677, 5611, 31136, 21309, 4746, 31125, 694, 1033, 653, 8808, 754, 650, 16814, 677, 5621, 1412, 4514, 678, 2835, 677, 31106, 53, 60, 57, 59, 79, 1057, 3142, 656, 16814, 772, 656, 15824, 4305, 31125, 680, 2384, 772, 656, 9592, 1161, 8480, 13550, 807, 31125, 1238, 742, 11135, 2521, 656, 1226, 679, 8431, 3392, 677, 4816, 8946, 79, 1057, 982, 4251, 650, 1697, 679, 3594, 31125, 1516, 776, 2835, 2409, 679, 7782, 1620, 762, 53, 60, 57, 60, 1098, 776, 8753, 2542, 17655, 762, 53, 60, 58, 53, 1098, 680, 776, 1127, 1596, 658, 2542, 17655, 762, 53, 60, 58, 54, 31145, 79, 5, 31131, 1033, 653, 796, 650, 2427, 23747, 679, 656, 3681, 2024, 751, 19422, 2790, 728, 31125, 680, 796, 650, 2736, 679, 656, 1625, 4859, 31155, 31114, 7284, 79, 1057, 796, 982, 650, 2736, 679, 656, 15824, 4305, 31125, 656, 1996, 1179, 4302, 784, 31125, 680, 656, 751, 19422, 4305, 79, 1057, 4357, 677, 2835, 677, 31106, 53, 61, 52, 56, 79, 5, 31131, 1033, 653, 796, 4204, 677, 656, 6730, 679, 5136, 942, 31124, 31136, 31115, 16987, 31136, 31133, 908, 31107, 22542, 31125, 677, 656, 1664, 2049, 679, 703, 667, 1024, 31125, 4746, 79, 1057, 796, 656, 3652, 679, 650, 675, 3034, 31125, 680, 796, 18735, 677, 656]
```

</details>


* Run several requests at the same time

```bash
echo '{"text_input": "What is machine learning?", "max_tokens": 20, "bad_words": "", "stop_words": "", "pad_id": 2, "end_id": 2}' > tmp.txt
printf '%s\n' {1..20} | xargs -I % -P 20 curl -X POST localhost:8000/v2/models/ensemble/generate -d @tmp.txt
```
