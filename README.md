# Tekit Backend
The Triton backend for Tekit.

## Usage

### Launch the backend *within Docker*

```bash
# 1. Pull the docker image
nvidia-docker run -it --rm -e LOCAL_USER_ID=`id -u ${USER}` --shm-size=2g -v <your/path>:<your/path> <image> bash
# <image> could be gitlab-master.nvidia.com:5005/ftp/tekit/triton:tot-ln

# 2. install tekit and tritonserver

pip install tekit --extra-index-url https://__token__:_un3_XBz1zhfMHp2mMLq@gitlab-master.nvidia.com/api/v4/projects/68442/packages/pypi/simple
pip install --extra-index-url https://pypi.ngc.nvidia.com regex fire tritonclient[all]

# 3. Modify parameters[engine_dir] in all_models/gpt/config.pbtxt to the path of built engines.

# 4. Launch triton server
python scripts/launch_triton_server.py --world_size=2
```

### Launch the backend *with Slurm*
tekit_triton.sub

```bash
#!/bin/bash
#SBATCH -o logs/tekit.out
#SBATCH -e logs/tekit.error
#SBATCH -J gpu-comparch-ftp:mgmn
#SBATCH -A gpu-comparch
#SBATCH -p luna
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --time=00:30:00

sudo nvidia-smi -lgc 1410,1410

srun --mpi=pmix --container-image gitlab-master.nvidia.com/ftp/tekit/triton:tot-ln --container-mounts /home/kevxie/lustre/workspace/:/workspace/ --container-workdir /workspace/tekit --output logs/tekit_%t.out --error logs/tekit_%t.error bash /workspace/tekit_triton.sh
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
python examples/gpt/client.py

perf_analyzer -m gpt --concurrency-range 1:4 -u 'localhost:8000' 2>&1 | tee triton_http_perf.log
perf_analyzer -m gpt --concurrency-range 1:4 -u 'localhost:8001' -i grpc 2>&1 | tee triton_grpc_perf.log
```
