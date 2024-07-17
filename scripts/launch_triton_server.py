import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size',
                        type=int,
                        default=1,
                        help='world size, only support tensor parallelism now')
    parser.add_argument(
        '--tritonserver',
        type=str,
        help='path to the tritonserver exe',
        default='/opt/tritonserver/bin/tritonserver',
    )
    parser.add_argument(
        '--grpc_port',
        type=str,
        help='tritonserver grpc port',
        default='8001',
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
        '--force',
        '-f',
        action='store_true',
        help='launch tritonserver regardless of other instances running')
    parser.add_argument('--log',
                        action='store_true',
                        help='log triton server stats into log_file')
    parser.add_argument(
        '--log-file',
        type=str,
        help='path to triton log gile',
        default='triton_log.txt',
    )

    path = str(Path(__file__).parent.absolute()) + '/../all_models/gpt'
    parser.add_argument('--model_repo', type=str, default=path)

    parser.add_argument(
        '--tensorrt_llm_model_name',
        type=str,
        help=
        'Name(s) of the tensorrt_llm Triton model in the repo. Use comma to separate if multiple model names',
        default='tensorrt_llm',
    )

    parser.add_argument(
        '--multi-model',
        action='store_true',
        help=
        'Enable support for multiple TRT-LLM models in the Triton model repository'
    )

    parser.add_argument(
        '--oversubscribe',
        action='store_true',
        help=
        'Append --oversubscribe to the mpirun command. Mainly for SLURM MPI usecases.'
    )

    return parser.parse_args()


def get_cmd(world_size, tritonserver, grpc_port, http_port, metrics_port,
            model_repo, log, log_file, tensorrt_llm_model_name, oversubscribe):
    cmd = ['mpirun', '--allow-run-as-root']
    if oversubscribe:
        cmd += ['--oversubscribe']
    for i in range(world_size):
        cmd += ['-n', '1', tritonserver, f'--model-repository={model_repo}']
        if log and (i == 0):
            cmd += ['--log-verbose=3', f'--log-file={log_file}']
        # If rank is not 0, skip loading of models other than `tensorrt_llm_model_name`
        if (i != 0):
            cmd += ['--model-control-mode=explicit']
            model_names = tensorrt_llm_model_name.split(',')
            for name in model_names:
                cmd += [f'--load-model={name}']
        cmd += [
            f'--grpc-port={grpc_port}', f'--http-port={http_port}',
            f'--metrics-port={metrics_port}', '--disable-auto-complete-config',
            f'--backend-config=python,shm-region-prefix-name=prefix{i}_', ':'
        ]
    return cmd


if __name__ == '__main__':
    args = parse_arguments()
    res = subprocess.run(['pgrep', '-r', 'R', 'tritonserver'],
                         capture_output=True,
                         encoding='utf-8')
    if res.stdout:
        pids = res.stdout.replace('\n', ' ').rstrip()
        msg = f'tritonserver process(es) already found with PID(s): {pids}.\n\tUse `kill {pids}` to stop them.'
        if args.force:
            print(msg, file=sys.stderr)
        else:
            raise RuntimeError(msg + ' Or use --force.')
    cmd = get_cmd(int(args.world_size), args.tritonserver, args.grpc_port,
                  args.http_port, args.metrics_port, args.model_repo, args.log,
                  args.log_file, args.tensorrt_llm_model_name,
                  args.oversubscribe)
    env = os.environ.copy()
    if args.multi_model:
        assert args.world_size == 1, 'World size must be 1 when using multi-model. Processes will be spawned automatically to run the multi-GPU models'
        env['TRTLLM_ORCHESTRATOR'] = '1'
    subprocess.Popen(cmd, env=env)
