import argparse
import subprocess
import sys
import os


def path_to_models(model):

    if model.startswith('gemini'):
        prefix = 'gemini'
    else:
        prefix = model
    PATH_TO_MODELS = {
        'gemini': './models/Genimi/run.py',
        'qwen2.5_omni': './models/Qwen2.5-Omni/run.py',
        'echoink': './models/EchoInk/run.py',
        'vita': './models/VITA/run.py',
    }
    
    return PATH_TO_MODELS.get(prefix, None)

def main(args):
    script_path = path_to_models(args.model)
    if not script_path:
        raise ValueError(f"Model {args.model} is not defined in PATH_TO_MODELS.")
    
    # if os.path.exists(os.path.join(args.save_dir, args.model, args.task_name+'.json')):
    #     print(f"Results for {args.model} on {args.task_name} already exist.")
    #     return
    
    if not args.reason:
        subprocess_args = [
            sys.executable,
            script_path,
            '--task_name', args.task_name,
            '--root_dir', args.root_dir,
            '--sample', str(args.sample),
            '--save_dir', os.path.join(args.save_dir.rstrip('/')+'_mini_benchmark' if args.mini_benchmark else args.save_dir, args.model)
        ]
        
        subprocess.run(subprocess_args, check=True)
        
    else:
        subprocess_args = [
            sys.executable,
            script_path,
            '--task_name', args.task_name,
            '--root_dir', args.root_dir,
            '--sample', str(args.sample),
            '--save_dir', os.path.join(args.save_dir.rstrip('/')+'_mini_benchmark' if args.mini_benchmark else args.save_dir, args.model),
            '--reason', 'True'
        ]
        
        subprocess.run(subprocess_args, check=True)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='# function')
    parser.add_argument('--root_dir', help='Root directory of AudioBench tasks', default='/home/xwang378/scratch/2025/AudioBench/benchmark/tasks/')
    parser.add_argument('--task_name', help='Name of the task to run', required=True)
    parser.add_argument('--sample', type=int, default=-1, help='Number of samples to process, if -1, run all samples')
    parser.add_argument('--mini_benchmark', action='store_true', default=False, help='Whether to run the mini benchmark')
    parser.add_argument('--save_dir', help='Directory to save results', default='/home/xwang378/scratch/2025/AudioBench/benchmark/results/')
    parser.add_argument('--reason', type=bool, default=False, help='Whether to run the reason script')
    args = parser.parse_args()
    main(args)