import argparse
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
        '--force',
        '-f',
        action='store_true',
        help='launch tritonserver regardless of other instances running')
    parser.add_argument('--log',
                        action='store_true',
                        help='log triton server stats into log_file')
    parser.add_argument(
        '--log_file',
        type=str,
        help='path to triton log gile',
        default='triton_log.txt',
    )

    path = str(Path(__file__).parent.absolute()) + '/../all_models/gpt'
    parser.add_argument('--model_repo', type=str, default=path)
    return parser.parse_args()


def get_cmd(world_size, tritonserver, model_repo, log, log_file):
    cmd = ['mpirun', '--allow-run-as-root']
    for i in range(world_size):
        cmd += ['-n', '1', tritonserver]
        if log and (i == 0):
            cmd += [f' --log --log-file={log_file}']
        cmd += [
            f'--model-repository={model_repo}',
            '--disable-auto-complete-config',
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
    cmd = get_cmd(int(args.world_size), args.tritonserver, args.model_repo,
                  args.log, args.log_file)
    subprocess.Popen(cmd)
