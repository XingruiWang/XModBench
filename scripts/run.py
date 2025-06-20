import argparse
import subprocess
import sys
import os


PATH_TO_MODELS = {
    'gemini': './models/Genimi/run.py',

}

def main(args):
    script_path = PATH_TO_MODELS.get(args.model)
    if not script_path:
        raise ValueError(f"Model {args.model} is not defined in PATH_TO_MODELS.")

    subprocess_args = [
        sys.executable,
        script_path,
        '--task_name', args.task_name,
        '--root_dir', args.root_dir,
        '--sample', str(args.sample),
        '--save_dir', os.path.join(args.save_dir, args.model)
    ]
    
    subprocess.run(subprocess_args, check=True)
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='# function')
    parser.add_argument('--root_dir', help='Root directory of AudioBench tasks', default='/home/xwang378/scratch/2025/AudioBench/benchmark/tasks/')
    parser.add_argument('--task_name', help='Name of the task to run', required=True)
    parser.add_argument('--sample', type=int, default=1, help='Number of samples to process')
    parser.add_argument('--save_dir', help='Directory to save results', default='/home/xwang378/scratch/2025/AudioBench/benchmark/results/')
    
    args = parser.parse_args()
    main(args)