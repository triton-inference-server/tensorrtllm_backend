# tensorrt-llm-example
docker build --target base-trtllm -t trt-example .

docker run --rm -it -p 80:80 --shm-size=10g --ulimit memlock=-1 --ulimit stack=67108864 --gpus '"device=4,5,6,7"' -e HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN -e HUGGINGFACE_HUB_CACHE=/data/trt-data -v /data:/data  trt-example:latest --world_size=4 --model_id=DeepInfra/Llama-2-70b-chat-hf-trt-fp8 --revision=5de4d5c03ffd13b8ac34bf50fb2e797f4d9be93e --tokenizer_model_id=DeepInfra/Llama-2-70b-chat-tokenizer --tokenizer_revision=f88981891fea1e38150df966c833e6d1e7e798f4 --http_port=80
