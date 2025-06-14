import argparse
import os
import numpy as np
import torch
import gc
from utils import load_model
from data_prepare import LoadDataset
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from cal_KFAC import cal_ihvp, cal_grad, cal_influence
import warnings
import datetime
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only=False.*")


def setup_distributed():
    torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=180000))
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def main(args):
    local_rank = setup_distributed()
    
    # load model
    model, tokenizer = load_model(args.model_path)

    train_sample_rate = 0.5
    val_sample_rate = 1.0
    # calculate KFAC
    train_dataset = LoadDataset(all_file_paths=args.full_train,
                                tokenizer=tokenizer,
                                max_seq_length=32768,
                                sample_percentage=train_sample_rate)
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=8, drop_last=True)
    # cal_ihvp(train_dataloader, model, args.save_path, args, local_rank) #注释此行以避免重复计算
    del train_dataloader
    del train_sampler
    del train_dataset
    torch.cuda.empty_cache()
    gc.collect()

    per_val_list = []
    # calculate validation gradient
    validation_dataset = LoadDataset(all_file_paths=args.validation_path,
                                    tokenizer=tokenizer,
                                    max_seq_length=32768,
                                    sample_percentage=val_sample_rate)
    validation_sampler = DistributedSampler(validation_dataset, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, sampler=validation_sampler, batch_size=8)
    # validation_grad_path = cal_grad(validation_dataloader, model, args.save_path + f"/val_avg_grad", args, local_rank) #注释此行以避免重复计算
    validation_grad_path=args.save_path + f"/val_avg_grad.pt"
    per_val_list.append(validation_grad_path)
    del validation_dataloader
    del validation_sampler
    del validation_dataset
    gc.collect()

    for subset in list_subdirectories(args.validation_path):
        subset_path = os.path.join(args.validation_path, subset)
        if os.path.exists(subset_path):
            dataset = LoadDataset(all_file_paths=subset_path,
                                  tokenizer=tokenizer,
                                  max_seq_length=32768, #max_length改
                                  sample_percentage=val_sample_rate)
            sampler = DistributedSampler(dataset, shuffle=True)
            dataloader = DataLoader(dataset, sampler=sampler, batch_size=8)
            # subset_grad_path = cal_grad(dataloader, model, args.save_path + f"/val_{subset}_grad", args, local_rank) #注释此行以避免重复计算
            subset_grad_path = args.save_path + f"/val_{subset}_grad.pt"
            per_val_list.append(subset_grad_path)
            # release memory
            del dataloader
            del sampler
            del dataset
            torch.cuda.empty_cache()
            gc.collect()
    
    if local_rank == 0:
        print(per_val_list)
    
    
    per_train_list = []
    
    for i, subset in enumerate(args.sub_train):
        subset_path = os.path.join(args.full_train, subset)
        if os.path.exists(subset_path):
            # 获取该子集中的所有json/jsonl文件
            subset_files = []
            for filename in os.listdir(subset_path):
                if filename.endswith('.json') or filename.endswith('.jsonl'):
                    file_path = os.path.join(subset_path, filename)
                    subset_files.append(file_path)
            
            # 为每个文件分别计算梯度
            subset_grad_paths = []
            for j, file_path in enumerate(subset_files):
                # 创建只包含单个文件的临时目录路径（用于LoadDataset）
                temp_dir = os.path.dirname(file_path)
                temp_filename = os.path.basename(file_path)
                
                # 创建只包含当前文件的数据集
                dataset = LoadDataset(all_file_paths=temp_dir,
                                    tokenizer=tokenizer,
                                    max_seq_length=32768,
                                    sample_percentage=train_sample_rate)
                
                
                if len(dataset.data_indices) > 0:  # 确保文件中有数据
                    sampler = DistributedSampler(dataset, shuffle=True)
                    dataloader = DataLoader(dataset, sampler=sampler, batch_size=8)
                    
                    # 为每个文件生成唯一的保存路径
                    file_basename = os.path.splitext(temp_filename)[0]
                    grad_save_path = args.save_path + f"/train_{subset}_{file_basename}_grad"
                    # avg_grad_path = cal_grad(dataloader, model, grad_save_path, args, local_rank) #注释此行以避免重复计算
                    avg_grad_path = grad_save_path+".pt"
                    subset_grad_paths.append(avg_grad_path)
                    
                    # release memory
                    del dataloader
                    del sampler
                
                del dataset
                torch.cuda.empty_cache()
                gc.collect()
            
            # 将该子集的所有文件梯度路径添加到列表中
            per_train_list.extend(subset_grad_paths)
            
    # 在计算完所有训练梯度后，重新初始化final_result
    if local_rank == 0:
        print(f"Total training gradient files: {len(per_train_list)}")
    
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    
    # 修改这里：基于实际的训练文件数量初始化
    final_result = np.zeros((len(per_val_list), len(per_train_list)))
    
    for i in range(len(per_train_list)):
        for j in range(len(per_val_list)):
            influence_list = cal_influence(hessian_path = args.save_path,
                        train_grad_path = per_train_list[i],
                        validation_grad_path = per_val_list[j], 
                        local_rank = local_rank)
            total_sum = sum(influence_list)
            final_result[j, i] = total_sum.item()

            del influence_list
            torch.cuda.empty_cache()
            gc.collect()

    if local_rank == 0:
        print(final_result)
        df = pd.DataFrame(final_result, index=per_val_list, columns=per_train_list)
        df.to_csv(args.save_path + f"/influence.csv", index=True, header=True)

def list_subdirectories(directory):
    subdirectories = []
    for root, dirs, files in os.walk(directory):
        for dir_name in dirs:
            subdirectories.append(dir_name)
    return subdirectories

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=f"Path of YOUR base model")
    parser.add_argument("--full-train", type=str, default=f"Path of YOUR full train dataset")
    parser.add_argument("--validation-path", type=str, default=f"Path of YOUR validation dataset")
    parser.add_argument("--sub-train",  nargs='+', type=str, default=[f"Mathematics","Coding","bbh","Instruction","TrustAI"])
    parser.add_argument("--save-path", type=str, default=f"Path of YOUR save folder")
    parser.add_argument("--use-full-layer", type=bool, default=True)
    #nargs='+' 表示该参数需要一个或多个值，并且 argparse 会将这些值收集到一个列表中
    parser.add_argument("--target-layers", nargs='+', type=str, default=["model.layers.1.mlp.gate_proj", "model.layers.5.mlp.gate_proj", "model.layers.10.mlp.gate_proj", "model.layers.15.mlp.gate_proj",
    "model.layers.20.mlp.gate_proj", "model.layers.24.mlp.gate_proj", "model.layers.25.mlp.gate_proj", "model.layers.26.mlp.gate_proj", "model.layers.27.mlp.gate_proj", "model.layers.28.mlp.gate_proj"])
    parser.add_argument("--without-output", type=bool, default=True)
    parser.add_argument("--without-attention", type=bool, default=True)
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)
        print(f"Directory '{args.save_path}' created.")

    main(args)
    torch.distributed.destroy_process_group()