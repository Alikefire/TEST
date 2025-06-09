import subprocess
import torch
import time
# import cycle # No longer needed for cycling through single GPUs
import argparse
import os
import re
import random # Add this import
import signal
import psutil  # 需要安装: pip install psutil

def get_available_gpus():
    """Returns list of available GPU indices as a comma-separated string."""
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        return ""
    return ",".join(map(str, range(num_gpus)))

def get_checkpoint_list(model_path):
    """Get list of available checkpoint numbers from the model path"""
    checkpoints = []
    checkpoint_pattern = re.compile(r'checkpoint-(\d+)')
    
    if not os.path.isdir(model_path):
        print(f"Error: Model path {model_path} does not exist or is not a directory.")
        return []

    for item in os.listdir(model_path):
        match = checkpoint_pattern.match(item)
        if match and os.path.isdir(os.path.join(model_path, item)):
            checkpoints.append(int(match.group(1)))
    
    return sorted(checkpoints)

def kill_process_tree(pid):
    """递归杀死进程树中的所有进程"""
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        
        # 先杀死所有子进程
        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        
        # 等待子进程结束
        gone, alive = psutil.wait_procs(children, timeout=10)
        
        # 强制杀死仍然存活的进程
        for p in alive:
            try:
                p.kill()
            except psutil.NoSuchProcess:
                pass
        
        # 最后杀死父进程
        try:
            parent.terminate()
            parent.wait(timeout=10)
        except (psutil.NoSuchProcess, psutil.TimeoutExpired):
            try:
                parent.kill()
            except psutil.NoSuchProcess:
                pass
                
    except psutil.NoSuchProcess:
        pass

def run_checkpoints_with_all_gpus(model_path, config_file, checkpoint_list):
    available_gpu_ids_str = get_available_gpus()
    if not available_gpu_ids_str:
        raise RuntimeError("No GPUs available!")
    
    num_available_gpus = len(available_gpu_ids_str.split(','))
    print(f"Found {num_available_gpus} GPUs. Will use all of them ({available_gpu_ids_str}) for each checkpoint.")
    
    # Process all checkpoints sequentially, but each using all available GPUs
    for ckpt in checkpoint_list:
        print(f"Starting checkpoint {ckpt} using GPUs: {available_gpu_ids_str}")
        
        # Construct the command. 
        # The get_trajectories.py script needs to be able to handle multi-GPU execution, 
        # possibly via torchrun or by itself if it initializes distributed environment.
        # For simplicity, we'll assume get_trajectories.py can be launched directly 
        # and will respect CUDA_VISIBLE_DEVICES for FSDP or similar.
        # If get_trajectories.py needs to be launched with torchrun:
        # cmd_prefix = [
        #     "torchrun",
        #     f"--nproc_per_node={num_available_gpus}",
        # ]
        # else, if get_trajectories.py handles distributed setup internally:
        cmd_prefix = [
            "accelerate", "launch",
            "--num_processes", str(num_available_gpus),
            "--gpu_ids", available_gpu_ids_str,
            "--main_process_port", str(29500 + ckpt % 1000)  # 避免端口冲突，默认是29500端口
        ]

        cmd = cmd_prefix + [
            "/home/zdd/xx_help/S2L/get_trajectories.py",
            "--model_path", model_path,
        ]
        
        if config_file:
            cmd.extend(["--config_file", config_file])
        
        cmd.extend(["--ckpt", str(ckpt)])
        
        # Set CUDA_VISIBLE_DEVICES for the subprocess
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = available_gpu_ids_str
        
        print(f"Setting MASTER_PORT={str(29500 + ckpt % 1000)} for checkpoint {ckpt}")

        # 启动进程
        process = subprocess.Popen(cmd, env=env)
        
        try:
            # 等待进程完成
            process.wait()
            
            if process.returncode == 0:
                print(f"Checkpoint {ckpt} completed successfully using GPUs {available_gpu_ids_str}")
            else:
                print(f"Checkpoint {ckpt} failed using GPUs {available_gpu_ids_str} with return code {process.returncode}")
        
        except KeyboardInterrupt:
            print(f"Interrupted during checkpoint {ckpt}, cleaning up...")
            kill_process_tree(process.pid)
            raise
        
        finally:
            # 强制清理进程树
            print(f"Cleaning up process tree for checkpoint {ckpt}...")
            kill_process_tree(process.pid)
            
            # 额外等待时间确保资源释放
            time.sleep(5)
            
            # 清理CUDA缓存
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print(f"CUDA cache cleared for checkpoint {ckpt}")
            except Exception as e:
                print(f"Warning: Could not clear CUDA cache: {e}")

        time.sleep(1) # Small delay if needed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True,
                        help='Either local model directory (with config) or HuggingFace model path')
    parser.add_argument('--config_file', type=str, default=None,
                        help='Config file path for local models')
    parser.add_argument('--checkpoints', type=str, default='all',
                        help='Comma-separated list of checkpoint numbers, range in format start:end:step, or "all" to process all checkpoints')
    
    args = parser.parse_args()
    
    # Parse checkpoint list
    if args.checkpoints.lower() == 'all':
        checkpoint_list_to_process = get_checkpoint_list(args.model_path)
        if not checkpoint_list_to_process:
            print(f"No checkpoints found in {args.model_path}. Exiting.")
            exit(1)
        print(f"Found {len(checkpoint_list_to_process)} checkpoints: {checkpoint_list_to_process}")
    elif ':' in args.checkpoints:
        try:
            start, end, step = map(int, args.checkpoints.split(':'))
            checkpoint_list_to_process = list(range(start, end + 1, step))
        except ValueError:
            print("Invalid range format for checkpoints. Use start:end:step")
            exit(1)
    else:
        try:
            checkpoint_list_to_process = [int(x.strip()) for x in args.checkpoints.split(',')]
        except ValueError:
            print("Invalid comma-separated list for checkpoints. Ensure all are integers.")
            exit(1)
    
    run_checkpoints_with_all_gpus(args.model_path, args.config_file, checkpoint_list_to_process)
    
# Process all checkpoints found in the model directory:
# python run_distributed_trajectories.py --model_path /path/to/model --config_file config.yaml --checkpoints all

# For a comma-separated list of checkpoints:
# python run_distributed_trajectories.py --model_path /path/to/model --config_file config.yaml --checkpoints 1000,2000,3000,4000

# Or using a range:
# python run_distributed_trajectories.py --model_path /path/to/model --config_file config.yaml --checkpoints 1000:5000:1000