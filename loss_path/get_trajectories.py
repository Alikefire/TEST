import argparse
import torch
import yaml
from tqdm import tqdm
from utils import rank0_print, get_model, get_tokenizer, smart_tokenizer_and_embedding_resize, make_supervised_data_module
import os
import gc # 导入gc模块
from collections import Counter
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add distributed imports
import torch.distributed as dist
import torch.nn.parallel as dp
from torch.utils.data import DataLoader, DistributedSampler

def loss(data, model, rank, world_size, tokenizer=None):
    """compute last hidden states for a data_module"""
    # model is already on the correct device if wrapped by DDP
    model.eval()

    losses = []
    loss_num = 0

    train_dataset = data["train_dataset"]
    if world_size > 1:
        sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank) #此处添加drop_last=True以确保不会添加样本来保证不同进程样本数一致
        dataloader = DataLoader(train_dataset, batch_size=1, sampler=sampler)
    else:
        dataloader = DataLoader(train_dataset, batch_size=1) # For single GPU, no sampler needed

    # 收集所有索引
    all_indices = []
    with torch.no_grad():
        # Iterate through the dataloader with DistributedSampler
        for _, datapoint in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Rank {rank}"):
            original_idx = datapoint['original_index'].item() #.item() 将张量其转回Python标量
            all_indices.append(original_idx)
            # # 打印部分原始索引对应内容进行检查
            # if(original_idx in {261, 262, 2216, 2217}):
            #     #分别对应s1.1第262条，第263条，gsm8k-r1第235行，236行
            #     # 解码并打印input_ids对应的文本内容
            #     if(tokenizer is not None):
            #         # 修正：使用 datapoint 中的 input_ids
            #         input_ids = datapoint['input_ids'].squeeze()  # 移除batch维度
            #         decoded_text = tokenizer.decode(input_ids, skip_special_tokens=True)
            #         print(f"Text content: {decoded_text[:500]}...")  # 显示前500个字符
            #         print("-" * 80)  # 分隔线
            print(f"Rank {rank}, Sample {_}, Original index: {original_idx}")
            
            # Data is already on CPU, move to GPU
            # Move to specific GPU based on rank
            input_ids = datapoint["input_ids"].cuda(rank)
            labels = datapoint["labels"].cuda(rank)
            
            # Ensure input_ids and labels have a batch dimension if they don't already
            # This handles cases where batch_size=1 and DataLoader might drop the batch dim
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            if labels.dim() == 1:
                labels = labels.unsqueeze(0)
            #调试input_ids和labels的形状和数据类型
            # print(f"Rank {rank}: input_ids shape: {input_ids.shape}, dtype: {input_ids.dtype}, device: {input_ids.device}")
            # print(f"Rank {rank}: labels shape: {labels.shape}, dtype: {labels.dtype}, device: {labels.device}")
            # # Optional: print tensor content for a few samples
            # if _ < 5: # Print for first 5 samples
            #     print(f"Rank {rank}: input_ids: {input_ids}")
            #     print(f"Rank {rank}: labels: {labels}")
            
            # Model is DDP wrapped, forward pass handles parallelism
            result = model(input_ids=input_ids, labels=labels, return_dict=True)
            current_loss = result.loss # 将loss赋给一个局部变量
            # report progress only on rank 0
            if rank == 0 and (_ == 1 or (_ != 0 and _ % 10000 == 0)):
                rank0_print(f"***** Predict-Progress -- {_} DONE !")
            losses.append(current_loss.detach().cpu())
            loss_num += 1

            # 清理不再需要的张量引用
            del input_ids
            del labels
            del result
            del current_loss
            
            # 执行垃圾回收和清空CUDA缓存
            gc.collect()
            torch.cuda.empty_cache()

        rank0_print(f"***** Rank {rank}: Processed {loss_num} samples.")

    # Gather losses from all ranks
    if world_size > 1:
        # 收集所有进程的索引
        gathered_indices = [None for _ in range(world_size)]
        dist.gather_object(all_indices, gathered_indices if rank == 0 else None, dst=0)

        gathered_losses = [None for _ in range(world_size)]
        # Gather objects to rank 0
        dist.gather_object(losses, gathered_losses if rank == 0 else None, dst=0)

        if rank == 0:
            # 合并所有索引
            all_indices_combined = [idx for sublist in gathered_indices for idx in sublist]
            # 合并所有损失
            all_losses_combined = [item for sublist in gathered_losses for item in sublist]

            # 将 original_index 和对应的 loss 值配对
            indexed_losses = []
            for i in range(len(all_indices_combined)):
                indexed_losses.append((all_indices_combined[i], all_losses_combined[i]))

            # 根据 original_index 进行排序
            indexed_losses.sort(key=lambda x: x[0])

            # 提取排序后的损失值
            sorted_losses = [loss_val for idx, loss_val in indexed_losses]

            # 检查重复，需要在主进程中完成，而不是每一个进程单独检查
            # 在 get_trajectories.py 中
            duplicates = [idx for idx, count in Counter(all_indices_combined).items() if count > 1]
            
            # 保存到文件（仅在有重复项时保存）
            if duplicates:  # 添加非空判断
                import json
                with open('duplicates_config.json', 'w') as f:
                    json.dump({'duplicates': duplicates}, f)
                print(f"Global duplicate indices: {duplicates}")
            else:
                print("No duplicate indices found.")

            # Flatten the list of lists
            rank0_print(f"***** All loss trajectories = {len(sorted_losses)}")
            return sorted_losses
        else:
            return None # Only rank 0 returns the combined losses
    else:
        return losses # In single GPU mode, just return the losses directly

def main(model_path, config_file=None, ckpt=-1, data_format="alpaca", preprocess_numworkers=1):
    # Initialize distributed environment
    rank = 0
    world_size = 1
    initialized_dist = False # Flag to track if dist.init_process_group was called

    if torch.cuda.is_available():
        if "WORLD_SIZE" in os.environ:
            world_size = int(os.environ["WORLD_SIZE"])
        if "RANK" in os.environ:
            rank = int(os.environ["RANK"])

        if world_size > 1:
            # Using nccl for GPU communication
            if not dist.is_initialized(): # Check if not already initialized
                dist.init_process_group(backend='nccl')
                initialized_dist = True # Set flag to true
            rank = dist.get_rank()
            world_size = dist.get_world_size()

        # Set device for the current process
        torch.cuda.set_device(rank)
    else:
        rank0_print("CUDA is not available. Running on CPU.")
        # If CUDA is not available, ensure rank and world_size are 0 and 1 respectively
        rank = 0
        world_size = 1

    if config_file:
        # Local model path logic
        with open(config_file, 'r') as f:
            args = yaml.full_load(f)
        if rank == 0:
            rank0_print('Configuration loaded!')
            rank0_print(yaml.dump(args, sort_keys=False, default_flow_style=False))

        args["output_dir_root"] = f"{args['result_dir_name']}"

        if ckpt == -1:
            model_path = args["output_dir_root"]+f"/"
        else:
            model_path = args["output_dir_root"]+f"/checkpoint-{ckpt}"

        loss_file = f"{model_path}/losses.pt"
    else:
        # HuggingFace model path logic
        args = {
            "cache_dir": None,
            "model_max_length": 2048,  # You might want to make this configurable
            "model_name_or_path": model_path
        }
        if ckpt != -1:
            model_path = f"{model_path}@{ckpt}"

        # Create a default output directory for HF models only on rank 0
        if rank == 0:
            os.makedirs("hf_outputs", exist_ok=True)
        loss_file = f"hf_outputs/{model_path.replace('/', '_')}_losses.pt"

    # Ensure all processes are ready before checking/saving file
    if world_size > 1:
        dist.barrier()

    # Only rank 0 checks if the file exists to avoid race conditions
    if rank == 0 and os.path.exists(loss_file):
        rank0_print(f"***** Losses have already existed at {loss_file}!")
        # Depending on desired behavior, you might want to exit here
        # dist.destroy_process_group()
        # return #如果需要losses.py文件存在不想再计算losses.py文件，就取消注释这行

    rank0_print(f"***=====================================================================================================")
    rank0_print(f"***** Checkpoint {ckpt} =====================================================================================================")
    model = get_model(model_name_or_path=model_path, cache_dir=args["cache_dir"])
    rank0_print(f'***** Model loaded from {model_path}!')

    # 将模型移动到当前设备的正确位置
    if torch.cuda.is_available():
        model.to(rank)
    
    # 将模型包装在 DistributedDataParallel 中
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    #get_tokenizer，换成qwen相关
    tokenizer, special_tokens_dict = get_tokenizer(model_name_or_path=args["model_name_or_path"], cache_dir=args["cache_dir"], model_max_length=args["model_max_length"],)
    rank0_print(f'***** Tokenizer initilized!')
    
    # 修改此处：如果模型是DDP包装的，则传递model.module
    if world_size > 1 and isinstance(model, torch.nn.parallel.DistributedDataParallel):
        tokenizer, _ = smart_tokenizer_and_embedding_resize(special_tokens_dict=special_tokens_dict, 
                                                            tokenizer=tokenizer, 
                                                            model=model.module)  # fix tokenizer's special_token_maps
    else:
        tokenizer, _ = smart_tokenizer_and_embedding_resize(special_tokens_dict=special_tokens_dict, 
                                                            tokenizer=tokenizer, 
                                                            model=model)  # fix tokenizer's special_token_maps
    rank0_print(f'***** smart_tokenizer_and_embedding_resize done!')
    full_data_path_str = args["full_data_path"] # example: "path1,path2,path3"
    if full_data_path_str:
        # 按逗号分割字符串，并去除每个路径周围可能存在的空格
        actual_data_paths_list = [path.strip() for path in full_data_path_str.split(',')]
    else:
        actual_data_paths_list = []
    all_data = make_supervised_data_module(tokenizer=tokenizer, data_path=actual_data_paths_list, data_format=data_format, preprocess_numworkers=preprocess_numworkers)

    mean_entropies_all = loss(data=all_data, model=model, rank=rank, world_size=world_size, tokenizer=tokenizer)
    # Ensure all processes are done before saving and destroying group
    if initialized_dist:
        dist.barrier() # Ensure all processes have finished loss computation

    if rank == 0: # Only rank 0 saves the file
        torch.save(mean_entropies_all, loss_file)
        print(f"***** Losses saved to {loss_file}")
    
    # Clean up distributed environment if it was initialized by this process
    if initialized_dist:
        dist.destroy_process_group()
        rank0_print("***** Distributed process group destroyed.")
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=False, default=None,
                        help='Config file path for local models')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Either local model directory (with config) or HuggingFace model path')
    parser.add_argument('--ckpt', type=int, default=-1,)
    parser.add_argument('--data_format', type=str, default="alpaca",help='读取的数据格式包含MathInstructh、alpaca、sharegpt,可以为列表或者字符串')
    parser.add_argument('--preprocess_numworkers', type=int, default=8 ,help='Accelerate data preprocess when processes huge dataset')
    args = parser.parse_args()
    
    main(model_path=args.model_path, config_file=args.config_file, ckpt=args.ckpt, data_format=args.data_format, preprocess_numworkers=args.preprocess_numworkers)