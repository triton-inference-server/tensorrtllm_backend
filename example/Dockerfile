# syntax = edrevo/dockerfile-plus

# Newer version of base container can be found at https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver/tags
FROM nvcr.io/nvidia/tritonserver:23.10-trtllm-python-py3 AS base-trtllm

INCLUDE+ ../dockerfile/Dockerfile.trt_llm_backend

# install huggingface
RUN pip install huggingface_hub
RUN pip install hf_transfer
RUN env HF_HUB_ENABLE_HF_TRANSFER=1

# copy files that will be used for running triton server
COPY launch_triton_server.py .
COPY hub.py .

ENTRYPOINT ["python3", "launch_triton_server.py"]
