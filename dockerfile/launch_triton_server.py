# The following code is adapted from the tensorrtllm_backend (https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/scripts/launch_triton_server.py)
# License: Apache License, Version 2.0 (https://www.apache.org/licenses/LICENSE-2.0)
import argparse
import subprocess
import os
import sys
import errno
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.utils import (
    LocalEntryNotFoundError,
    EntryNotFoundError,
    RevisionNotFoundError,  # Import here to ease try/except in other part of the lib
)
from hub import download_weights, weight_files, weight_hub_files


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--world_size',
        type=int,
        default=4,
        help='world size, only support tensor parallelism now')
    parser.add_argument(
        '--tritonserver',
        type=str,
        help='path to the tritonserver exe',
        default='/opt/tritonserver/bin/tritonserver',
    )
    parser.add_argument(
        '--force',
        '-f',
        action='store_true',
        help='launch tritonserver regardless of other instances running')
    parser.add_argument(
        '--log',
        action='store_true',
        help='log triton server stats into log_file')
    parser.add_argument(
        '--log-file',
        type=str,
        help='path to triton log gile',
        default='triton_log.txt',
    )
    parser.add_argument(
        '--model_id',
        type=str,
        help='model from the huggingface',
        default='DeepInfra/Llama-2-70b-chat-hf-trt-fp8',
    )
    parser.add_argument(
        '--revision',
        type=str,
        help='revision of the model_id',
        default='5de4d5c03ffd13b8ac34bf50fb2e797f4d9be93e',
    )
    parser.add_argument(
        '--tokenizer_model_id',
        type=str,
        help='tokenizer model from the huggingface',
        default='DeepInfra/Llama-2-70b-chat-tokenizer',
    )
    parser.add_argument(
        '--tokenizer_revision',
        type=str,
        help='revision of the tokenizer_model_id',
        default='f88981891fea1e38150df966c833e6d1e7e798f4',
    )
    parser.add_argument(
        '--http_port',
        type=str,
        help='tritonserver http port',
        default='8000',
    )
    parser.add_argument(
        '--metrics_port',
        type=str,
        help='tritonserver metrics port',
        default='8002',
    )
    parser.add_argument(
        '--grpc_port',
        type=str,
        help='tritonserver grpc port',
        default='8001',
    )

    return parser.parse_args()


def get_cmd(world_size, tritonserver, model_repo, log, log_file, http_port, metrics_port, grpc_port):
    cmd = ['mpirun', '--allow-run-as-root']
    for i in range(world_size):
        cmd += ['-n', '1', tritonserver]
        if log and (i == 0):
            cmd += ['--log-verbose=3', f'--log-file={log_file}']
        cmd += [
            f'--model-repository={model_repo}',
            f'--http-port={http_port}',
            f'--metrics-port={metrics_port}',
            f'--grpc-port={grpc_port}',
            '--disable-auto-complete-config',
            f'--backend-config=python,shm-region-prefix-name=prefix{i}_', ':'
        ]
    return cmd


def download_hf_model_into(model_id, revision):
    extension = ""
    try:
        weight_files(model_id, revision, extension)
        print("Files are already present on the host. " "Skipping download.")
        return
        # Local files not found
    except (LocalEntryNotFoundError, FileNotFoundError):
        pass

    is_local_model = (Path(model_id).exists() and Path(model_id).is_dir()) or os.getenv(
        "WEIGHTS_CACHE_OVERRIDE", None
    ) is not None

    if not is_local_model:
        # Try to download weights from the hub
        try:
            filenames = weight_hub_files(model_id, revision, extension)
            print(filenames)
            download_weights(filenames, model_id, revision)
            # Successfully downloaded weights
            return

        # No weights found on the hub with this extension
        except EntryNotFoundError as e:
            # Check if we want to automatically convert to safetensors or if we can use .bin weights instead
            raise e


def run_cmd(cmd):
    try:
        # Spawn a new process using subprocess
        subprocess.run(cmd, check=True)
        # If the command succeeds, the following lines won't be executed
        print("The command failed.")
        os._exit(1)
    except subprocess.CalledProcessError as e:
        # If the command fails, exit the current process with the same exit code
        os._exit(e.returncode)


def symlink(link, folder):
    if os.path.exists(link):
        os.remove(link)
    os.symlink(folder, link)


def get_cached_dir(model_id, revision):
    folder = f'/data/trt-data/models--{model_id.replace("/", "--")}/snapshots/{revision}'
    return folder


def replace_placeholders(folder, tokenizer_repo, model_repo):
    d = {
        'preprocessing': ["${tokenizer_dir}", tokenizer_repo],
        'postprocessing': ["${tokenizer_dir}", tokenizer_repo],
        'tensorrt_llm': ["${gpt_model_path}", f'{model_repo}/tensorrt_llm/1'],
    }

    for k, v in d.items():
        path = f'{folder}/{k}/config.pbtxt'
        replace_string_in_file(path, v[0], v[1])


def replace_string_in_file(file_path, old_string, new_string):
    with open(file_path, 'r') as file:
        file_content = file.read()
    modified_content = file_content.replace(old_string, new_string)
    with open(file_path, 'w') as file:
        file.write(modified_content)
    print(f'File updated:{file_path}')


if __name__ == '__main__':
    args = parse_arguments()
    res = subprocess.run(['pgrep', '-r', 'R', 'tritonserver'],
                         capture_output=True,
                         encoding='utf-8')

    download_hf_model_into(args.tokenizer_model_id, args.tokenizer_revision)
    download_hf_model_into(args.model_id, args.revision)
    replace_placeholders(
        get_cached_dir(args.model_id, args.revision),
        get_cached_dir(args.tokenizer_model_id, args.tokenizer_revision),
        get_cached_dir(args.model_id, args.revision),
    )

    if res.stdout:
        pids = res.stdout.replace('\n', ' ').rstrip()
        msg = f'tritonserver process(es) already found with PID(s): {pids}.\n\tUse `kill {pids}` to stop them.'
        if args.force:
            print(msg, file=sys.stderr)
        else:
            raise RuntimeError(msg + ' Or use --force.')
    cmd = get_cmd(int(args.world_size), args.tritonserver, get_cached_dir(args.model_id, args.revision),
                  args.log, args.log_file, args.http_port, args.metrics_port, args.grpc_port)
    run_cmd(cmd)
