# Instructions to run TRT-LLM in-flight batching Triton backend:

## Build TensorRT-LLM engine for inflight batching

To configure a Triton server that runs a model using TensorRT-LLM, it is needed to compile a TensorRT-LLM engine for that model.

For example, for LLaMA 7B, change to the `tensorrt_llm/examples/llama` directory:

```
cd tensorrt_llm/examples/llama
```
Prepare the checkpoint of the model by following the instructions [here](https://huggingface.co/docs/transformers/main/en/model_doc/llama) and store it in a model directory. Then, create the engine:

```
python build.py --model_dir ${model_directory} \
                --dtype bfloat16 \
                --use_gpt_attention_plugin bfloat16 \
                --use_inflight_batching \
                --paged_kv_cache \
                --remove_input_padding \
                --use_gemm_plugin bfloat16 \
                --output_dir engines/bf16/1-gpu/
```

To disable the support for in-flight batching (i.e. use the V1 batching mode), remove `--use_inflight_batching`.

Similarly, for a GPT model, change to `tensorrt_llm/examples/gpt` directory:
```
cd tensorrt_llm/examples/gpt

```
Prepare the model checkpoint following the instructions in the README file, store it in a model directory and build the TRT engine with:

```
python3 build.py --model_dir=${model_directory} \
                 --dtype float16 \
                 --use_inflight_batching \
                 --use_gpt_attention_plugin float16 \
                 --paged_kv_cache \
                 --use_gemm_plugin float16 \
                 --remove_input_padding \
                 --use_layernorm_plugin float16 \
                 --hidden_act gelu \
                 --output_dir=engines/fp16/1-gpu
```

## Create a model repository folder

First run:
```
rm -rf triton_model_repo
mkdir triton_model_repo
cp -R all_models/inflight_batcher_llm/* triton_model_repo
```

Then copy the TRT engine to `triton_model_repo/tensorrt_llm/1/`. For example for the LLaMA 7B example above, run:

```
cp -R tensorrt_llm/examples/llama/engines/bf16/1-gpu/ triton_model_repo/tensorrt_llm/1
```

For the GPT example above, run:
```
cp -R tensorrt_llm/examples/gpt/engines/fp16/1-gpu/ triton_model_repo/tensorrt_llm/1
```


Edit the `triton_model_repo/tensorrt_llm/config.pbtxt` file and replace `${decoupled_mode}` with `True` or `False`, and `${engine_dir}` with `/triton_model_repo/tensorrt_llm/1/1-gpu/` since the `triton_model_repo` folder created above will be mounted to `/triton_model_repo` in the Docker container. Decoupled mode must be set to true if using the streaming option from the client.


To use V1 batching, the `config.pbtxt` should have:
```
parameters: {
  key: "gpt_model_type"
  value: {
    string_value: "V1"
  }
}
```

For in-flight batching, use:
```
parameters: {
  key: "gpt_model_type"
  value: {
    string_value: "inflight_fused_batching"
  }
}
```

By default, in-flight batching will try to overlap the execution of batches of
requests. It may have a negative impact on performance when the number of
requests is too small. To disable that feature, set the `enable_trt_overlap`
parameter to `False` in the `config.pbtxt` file:

```
parameters: {
  key: "enable_trt_overlap"
  value: {
    string_value: "False"
  }
}
```

Or, equivalently, add `enable_trt_overlap:False` to the invocation of the
`fill_template.py` tool:

```bash
python3 tools/fill_template.py -i all_models/inflight_batcher_llm/tensorrt_llm/config.pbtxt "enable_trt_overlap:False"
```

## Launch the Triton server container using the model_repository you just created

```
docker run --rm -it --net host --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --gpus='"'device=0'"' -v $(pwd)/triton_model_repo:/triton_model_repo tritonserver:w_trt_llm_backend /bin/bash -c "tritonserver --model-repository=/triton_model_repo"
```

## Run the provided client to send a request

You can test the inflight batcher server with the provided reference python client as following:
```
python3 inflight_batcher_llm/client/inflight_batcher_llm_client.py --request-output-len 200
```

You can also stop the generation process early by using the `--stop-after-ms` option to send a stop request after a few milliseconds:

```
python inflight_batcher_llm_client.py --stop-after-ms 200 --request-output-len 200
```

You will find that the generation process is stopped early and therefore the number of generated tokens is lower than 200.

You can have a look at the client code to see how early stopping is achieved.

## Run the e2e/identity test to benchmark

### End to end test
End to end test script sends requests to deployed ensemble model.

Ensemble model is ensembled by three models: preprocessing, tensorrt_llm and postprocessing.
* preprocessing: Tokenizing, meaning the conversion from prompts(string) to input_ids(list of ints).
* tensorrt_llm: Inferencing.
* postprocessing: De-tokenizing, meaning the conversion from output_ids(list of ints) to outputs(string).

The end to end latency includes the total latency of the three parts of an ensemble model.

```
cd tools/inflight_batcher_llm
python3 end_to_end_test.py --dataset <dataset path>
```
Expected outputs
```
[INFO] Functionality test succeed.
[INFO] Warm up for benchmarking.
[INFO] Start benchmarking on 125 prompts.
[INFO] Total Latency: 11099.243 ms
```

### Identity test

Identity test script sends requests directly to deployed tensorrt_llm model, the identity test latency indicates the inference latency of TensorRT-LLM, not including the pre/post-processing latency which is usually handled by a third-party library such as HuggingFace.

Identity test can generate traffic from 2 sources.
1 - dataset (json file containning prompts and optional responses)
2 - token normal distribution (user specified input, output seqlen)

By default, the test uses exponential distrution to control arrival rate of requests. It can be changed to constant arrival time.

```
cd tools/inflight_batcher_llm
```
Example: Run dataset with 10 req/sec requested rate with provided tokenizer.
```
python3 identity_test.py -i grpc --request_rate 10 dataset --dataset <dataset path> --tokenizer_dir <> --tokenizer_type <>
```
Example: Generate I/O seqlen tokens with input normal distribution with mean_seqlen=128, stdev=10. Output normal distribution with mean_seqlen=20, stdev=2. Set stdev=0 to get constant seqlens.
```
python3 identity_test.py -i grpc --request_rate 10 token_norm_dist --input_mean 128 --input_stdev 5 --output_mean 20 --output_stdev 2 --num_requests 5000
```
Expected outputs
```
[INFO] Warm up for benchmarking.
[INFO] Start benchmarking on 5000 prompts.
[INFO] Total Latency: 26585.349 ms
[INFO] Total request latencies: 11569672.000999955 ms
+----------------------------+----------+
|            Stat            |  Value   |
+----------------------------+----------+
|        Requests/Sec        |  188.09  |
|       OP tokens/sec        | 3857.66  |
|     Avg. latency (ms)      | 2313.93  |
|      P99 latency (ms)      | 3624.95  |
|      P90 latency (ms)      | 3127.75  |
| Avg. IP tokens per request |  128.53  |
| Avg. OP tokens per request |  20.51   |
|     Total latency (ms)     | 26582.72 |
|       Total requests       | 5000.00  |
+----------------------------+----------+

```
*Please note that the expected outputs in that document are only for reference, specific performance numbers depend on the GPU you're using.*
