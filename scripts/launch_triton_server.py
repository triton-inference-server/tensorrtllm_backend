import argparse
import subprocess
from pathlib import Path


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size',
                        type=int,
                        default=1,
                        help='world size, only support tensor parallelism now')
    parser.add_argument('--tritonserver',
                        type=str,
                        default='/opt/tritonserver/bin/tritonserver')
    path = str(Path(__file__).parent.absolute()) + '/../all_models/gpt'
    parser.add_argument('--model_repo', type=str, default=path)
    return parser.parse_args()


def get_cmd(world_size, tritonserver, model_repo):
    cmd = ['mpirun', '--allow-run-as-root']
    for i in range(world_size):
        cmd += [
            '-n', '1', tritonserver, f'--model-repository={model_repo}',
            '--disable-auto-complete-config',
            f'--backend-config=python,shm-region-prefix-name=prefix{i}_', ':'
        ]
    return cmd


if __name__ == '__main__':
    args = parse_arguments()
    res = subprocess.run(['pgrep', 'tritonserver'],
                         capture_output=True,
                         encoding='utf-8')
    if res.stdout:
        pids = res.stdout.replace('\n', ' ').rstrip()
        raise RuntimeError(
            f'tritonserver process(es) already found with PID(s): {pids}.\n\tUse `kill {pids}` to stop them.'
        )
    cmd = get_cmd(int(args.world_size), args.tritonserver, args.model_repo)
    subprocess.Popen(cmd)
