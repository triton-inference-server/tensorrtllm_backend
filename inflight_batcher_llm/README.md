# Instructions to run TRT-LLM in-flight batching Triton backend:

## Build the Triton server image that includes the TRT-LLM in-flight batching backend:

From `tekit_backend` root folder:

```
docker build -f dockerfile/Dockerfile.trt_llm_backend -t tritonserver:23.04-py3_w_trt_llm_backend .
```

## Create a model repository folder

First run:
```
rm -rf triton_model_repo
mkdir triton_model_repo
cp -R all_models/inflight_batcher_llm/ triton_model_repo
```

Then copy the TRT model directory to `triton_model_repo/tensorrt_llm/1/`. For example if `<TRT_model_directory>`  is `tekit/cpp/tests/resources/models/rt_engine/gpt2/fp16-plugin-packed/1-gpu/` then run
```
cp -R tekit/cpp/tests/resources/models/rt_engine/gpt2/fp16-plugin-packed/1-gpu/ triton_model_repo/tensorrt_llm/1
```

Edit the `triton_model_repo/tensorrt_llm/config.pbtxt` file to set the path to the TRT model directory and the type of GPT model to use.


To use V1 batching, the `config.pbtxt` should have:
```
parameters: {
  key: "gpt_model_type"
  value: {
    string_value: "V1"
  }
}
parameters: {
  key: "gpt_model_path"
  value: {
    string_value: "/triton_model_repo/tensorrt_llm/1/1-gpu"
  }
}
```

For in-flight batching, use:
```
parameters: {
  key: "gpt_model_type"
  value: {
    string_value: "inflight_batching"
  }
}
parameters: {
  key: "gpt_model_path"
  value: {
    string_value: "/triton_model_repo/tensorrt_llm/1/1-gpu"
  }
}
```

## Launch the Triton server container using the model_repository you just created

```

docker run --rm -it --net host --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --gpus='"'device=0'"' -v $(pwd)/triton_model_repo:/triton_model_repo tritonserver:23.04-py3_w_trt_llm_backend /bin/bash -c "tritonserver --model-repository=/triton_model_repo"
```


## Run the provided client to send a request

```
python3 inflight_batcher_llm/client/inflight_batcher_llm_client.py
```
