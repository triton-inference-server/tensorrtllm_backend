# TensorRT-LLM Backend
The Triton backend for [TensorRT-LLM](https://docs.google.com/document/d/1g3_mL63pn5a72-eyMsL127jKedSaSdrv8fOh1Gcdgi8/edit).

## Usage

### Launch the backend *within Docker*

```bash
# 1. Pull the docker image
nvidia-docker run -it --rm -e LOCAL_USER_ID=`id -u ${USER}` --shm-size=2g -v <your/path>:<mount/path> <image> bash
# Recommend <image>: gitlab-master.nvidia.com:5005/ftp/tekit_backend/triton:23.04

# 2. Modify parameters in all_models/<model>/tekit/config.pbtxt

# 3. Launch triton server
python3 scripts/launch_triton_server.py --world_size=1 \
    --model_repo=all_models/<model>
```

### Launch the backend *within Slurm based clusters*
1. Prepare some scripts

`tensorrt_llm_triton.sub`
```bash
#!/bin/bash
#SBATCH -o logs/tensorrt_llm.out
#SBATCH -e logs/tensorrt_llm.error
#SBATCH -J gpu-comparch-ftp:mgmn
#SBATCH -A gpu-comparch
#SBATCH -p luna
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=00:30:00

sudo nvidia-smi -lgc 1410,1410

srun --mpi=pmix --container-image <image> \
    --container-mounts <your/path>:<mount/path> \
    --container-workdir <workdir> \
    --output logs/tensorrt_llm_%t.out \
    bash <workdir>/tensorrt_llm_triton.sh
```

`tensorrt_llm_triton.sh`
```
TRITONSERVER="/opt/tritonserver/bin/tritonserver"
MODEL_REPO="<workdir>/triton_backend/"

${TRITONSERVER} --model-repository=${MODEL_REPO} --backend-config=python,shm-region-prefix-name=prefix${SLURM_PROCID}_
```

2. Submit a Slurm job
```
sbatch tensorrt_llm_triton.sub
```

### Kill the backend

```bash
bash scripts/kill_triton_server.sh
```

## Examples

### GPT
```bash
cd tools/gpt/

# Download vocab and merge table for HF models
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt

python3 client.py \
    --text="Born in north-east France, Soyer trained as a" \
    --output_len=10

# Exmaple output:
# [INFO] Latency: 92.278 ms
# Input: Born in north-east France, Soyer trained as a
# Output:  chef and a cook at the local restaurant, La
```

## Test

```bash
cd tools/gpt/

# Identity test
python3 identity_test.py \
    --batch_size=8 --start_len=128 --output_len=20
# Results:
# [INFO] Batch size: 8, Start len: 8, Output len: 10
# [INFO] Latency: 70.782 ms
# [INFO] Throughput: 113.023 sentences / sec

# Benchmark using Perf Analyzer
python3 gen_input_data.py
perf_analyzer -m tensorrt_llm \
    -b 8 --input-data input_data.json \
    --concurrency-range 1:10:2 \
    -u 'localhost:8000'

# Results:
# Concurrency: 1, throughput: 99.9875 infer/sec, latency 79797 usec
# Concurrency: 3, throughput: 197.308 infer/sec, latency 121342 usec
# Concurrency: 5, throughput: 259.077 infer/sec, latency 153693 usec
# Concurrency: 7, throughput: 286.18 infer/sec, latency 195011 usec
# Concurrency: 9, throughput: 307.067 infer/sec, latency 233354 usec
```
