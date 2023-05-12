# Tekit Backend
The Triton backend for Tekit.

## Usage

### Launch the backend *within Docker*

```bash
# 1. Pull the docker image
nvidia-docker run -it --rm -e LOCAL_USER_ID=`id -u ${USER}` --shm-size=2g -v <your/path>:<mount/path> <image> bash
# Recommend <image>: gitlab-master.nvidia.com:5005/ftp/tekit_backend/triton:23.04

# 2. Modify parameters in all_models/xxx/tekit/config.pbtxt

# 3. Launch triton server
python3 scripts/launch_triton_server.py --world_size=1 \
    --model_repo=all_models/gpt
```

### Launch the backend *within Slurm based clusters*
tekit_triton.sub

```bash
#!/bin/bash
#SBATCH -o logs/tekit.out
#SBATCH -e logs/tekit.error
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
    --output logs/tekit_%t.out \
    bash <workdir>/tekit_triton.sh
```

tekit_triton.sh
```
export PYTHONPATH=/workspace/tekit
export CUDA_DEVICE_MAX_CONNECTIONS=1

WORK_DIR=${PYTHONPATH}
TRITONSERVER="/opt/tritonserver/bin/tritonserver"
MODEL_REPO="${WORK_DIR}/triton_backend/"

${TRITONSERVER} --model-repository=${MODEL_REPO} --backend-config=python,shm-region-prefix-name=prefix${SLURM_PROCID}_
```

### Kill the backend

```bash
bash scripts/kill_triton_server.sh
```

## Test

```bash
cd examples/gpt/
python3 client.py

perf_analyzer -m tekit --concurrency-range 1:4 -u 'localhost:8000' 2>&1 | tee triton_http_perf.log
perf_analyzer -m tekit --concurrency-range 1:4 -u 'localhost:8001' -i grpc 2>&1 | tee triton_grpc_perf.log
```
