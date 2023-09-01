# Instructions to run TRT-LLM in-flight batching Triton backend:

## Build TensorRT-LLM engine for inflight batching

To configure a Triton server that runs a model using TensorRT-LLM, it is needed to compile a TensorRT-LLM engine for that model.

For example, for LLaMA 7B, change to the `tekit/examples/llama` directory:

```
cd tekit/examples/llama
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

Similarly, for a GPT model, change to `tekit/examples/gpt` directory:
```
cd tekit/examples/gpt

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

## Build the Triton server image that includes the TRT-LLM in-flight batching backend:

From `tekit_backend` root folder:

```
docker build -f dockerfile/Dockerfile.trt_llm_backend -t tritonserver:w_trt_llm_backend .
```

## Create a model repository folder

First run:
```
rm -rf triton_model_repo
mkdir triton_model_repo
cp -R all_models/inflight_batcher_llm/ triton_model_repo
```

Then copy the TRT engine to `triton_model_repo/tensorrt_llm/1/`. For example for the LLaMA 7B example above, run:

```
cp -R tekit/examples/llama/engines/bf16/1-gpu/ triton_model_repo/tensorrt_llm/1
```

For the GPT example above, run:
```
cp -R tekit/examples/gpt/engines/fp16/1-gpu/ triton_model_repo/tensorrt_llm/1
```


Edit the `triton_model_repo/tensorrt_llm/config.pbtxt` file and replace `${decoupled_mode}` with `True` or `False`, and `${engine_dir}` with `/triton_model_repo/tensorrt_llm/1/` since the `triton_model_repo` folder created above will be mounted to `/triton_model_repo` in the Docker container. Decoupled mode must be set to true if using the streaming option from the client.


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
```
