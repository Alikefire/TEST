import torch
import numpy as np
import faiss
import matplotlib.pyplot as plt
import os
import glob
import json # Added for jload/load_jsonl simple implementation
from collections import Counter # Added for counting labels
import re

# Assuming datasets.load_dataset might be needed. If not available in this script's direct context,
# this part would need adjustment or to rely on a shared utility.
try:
    from datasets import load_dataset
except ImportError:
    print("Warning: 'datasets' library not found. Dataset loading for labels might be limited.")
    def load_dataset(path_or_name):
        # Dummy implementation if datasets is not available
        print(f"Warning: load_dataset called with dummy implementation for {path_or_name}")
        return {"train": []} 

# Simplified helper for loading json/jsonl for sample counting
# In a real scenario, these would come from utils.py or a robust library
def simple_load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def simple_jload(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Helper function to save data (add this near other helpers)
def save_data_to_file(data, file_path):
    """Saves data to a JSON or JSONL file."""
    # Ensure the directory exists
    dir_name = os.path.dirname(file_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print(f"Created directory for saving data: {dir_name}")

    if file_path.endswith(".jsonl"):
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        print(f"Saved {len(data)} items to JSONL: {file_path}")
    elif file_path.endswith(".json"):
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        print(f"Saved data to JSON: {file_path}")
    else:
        print(f"Warning: Unknown file type for saving: {file_path}. Data not saved.")


def load_dataset_info_from_paths(data_paths_list: list):
    """
    Loads dataset information (labels and names) from a list of data paths.
    Each sample gets an integer label corresponding to its source dataset.

    Args:
        data_paths_list (list): A list of paths to dataset files or Hugging Face dataset names.

    Returns:
        tuple: (all_sample_labels, unique_dataset_names)
               all_sample_labels: A list of integer labels for each sample, in order.
               unique_dataset_names: A list of unique dataset names corresponding to the integer labels.
    """
    all_sample_labels = []
    unique_dataset_names = []
    # 将 duplicated_sample_list 转换为 set，以便高效地进行查找和删除
    # 读取配置文件
    try:
        with open('duplicates_config.json', 'r') as f:
            config = json.load(f)
            duplicated_sample_set = set(config.get('duplicates', []))  # 默认为空列表
    except FileNotFoundError:
        duplicated_sample_set = set()  # 改为空集
    
    current_global_offset = 0 # 跟踪当前数据集在全局样本中的起始索引

    if not isinstance(data_paths_list, list):
        print(f"Warning: data_paths_list is not a list: {data_paths_list}. Treating as single item list.")
        data_paths_list = [data_paths_list]

    for path in data_paths_list:
        # 获取上一级目录的名称和文件名（不含扩展名）来构建唯一的名称
        parent_dir_name = os.path.basename(os.path.dirname(path))
        file_basename = os.path.basename(path).split('.')[0]
        dataset_name = f"{parent_dir_name}_{file_basename}"
        
        if dataset_name not in unique_dataset_names:
            unique_dataset_names.append(dataset_name)
        
        num_samples_from_file = 0 # 从文件中实际加载的样本数量
        try:
            if isinstance(path, str) and path.endswith(".jsonl"):
                data = simple_load_jsonl(path)
                num_samples_from_file = len(data)
            elif isinstance(path, str) and path.endswith(".json"):
                data = simple_jload(path)
                num_samples_from_file = len(data)
            elif isinstance(path, str): # Assuming Hugging Face dataset identifier
                # This part relies on the 'datasets' library
                hf_dataset = load_dataset(path, split='train') # Or appropriate split
                num_samples_from_file = len(hf_dataset)
            else:
                print(f"Warning: Unsupported data path format for label generation: {path}")
                continue # 跳过当前路径，处理下一个

            # 检查当前数据集是否包含重复样本，并调整 num_samples
            samples_to_add_for_duplicates = 0
            # 遍历 duplicated_sample_set 的副本，以便在迭代时安全地修改原集合
            for dup_idx in list(duplicated_sample_set):
                # 如果重复样本的全局索引落在当前数据集的范围内
                if current_global_offset <= dup_idx < current_global_offset + num_samples_from_file:
                    samples_to_add_for_duplicates += 1
                    duplicated_sample_set.remove(dup_idx) # 从集合中移除已处理的重复索引
                    print(f"  Warning: Duplicated sample index {dup_idx} found in {dataset_name}. Incrementing sample count.")
            
            # 最终的样本数量是文件加载数量加上因重复样本而增加的数量
            num_samples = num_samples_from_file + samples_to_add_for_duplicates

        except Exception as e:
            print(f"Error loading data from {path} for label generation: {e}. Skipping this source.")
            continue
        
        all_sample_labels.extend([unique_dataset_names.index(dataset_name)] * num_samples)
        print(f"  Loaded {num_samples} samples from {dataset_name} (label: {unique_dataset_names.index(dataset_name)})")
        current_global_offset += num_samples_from_file # 更新全局偏移量

    if not all_sample_labels:
        print("Warning: No dataset labels were generated. Check data paths and formats.")
        return [], []
        
    return all_sample_labels, unique_dataset_names


def load_losses_from_checkpoints(ref_model_path, num_loss_ckpts=-1, train_idx=None):
    """
    从指定的 checkpoint 目录加载 loss 数据。

    参数:
        ref_model_path (str): 包含各个 checkpoint 子目录的路径。
                              每个子目录应包含一个 'losses.pt' 文件。
        num_loss_ckpts (int): 要保留的 loss checkpoint 的数量。
                              如果为 -1，则使用所有找到的 checkpoints。
                              如果大于0且小于找到的checkpoint数量，则会进行下采样。
        train_idx (torch.Tensor, optional): 用于索引已加载 losses 的训练样本索引。
                                            如果为 None，则返回所有样本的 losses。

    返回:
        torch.Tensor: 加载和处理后的 loss 张量，形状为 (num_samples, num_features)。
                      如果发生错误或未找到 losses，则返回 None。
    """
    print(f"Loading losses from: {ref_model_path}")
    losses_list = []
    step_indices = [] # 新增：用于存储从文件名提取的 step index
    # 使用自定义排序key，确保checkpoint路径按数字顺序排序
    def natural_sort_key(s):
        # 提取字符串中的数字部分，并转换为整数进行比较
        return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', s)]

    checkpoint_paths = sorted(glob.glob(os.path.join(ref_model_path, '*')), key=natural_sort_key)

    if not checkpoint_paths:
        print(f"No checkpoint directories found in {ref_model_path}")
        return None, None # 修改：返回两个 None

    for ckpt_path in checkpoint_paths:
        loss_file = os.path.join(ckpt_path, "losses.pt")
        if os.path.isdir(ckpt_path) and os.path.exists(loss_file):
            print(f"  Loading losses from: {loss_file}")
            try:
                # 从 ckpt_path 提取 step index
                # 假设路径格式为 '.../checkpoint-STEP' 或 '.../STEP'
                basename = os.path.basename(ckpt_path)
                match = re.search(r'(\d+)$', basename) # 尝试匹配末尾的数字
                if not match and 'checkpoint-' in basename:
                    match = re.search(r'checkpoint-(\d+)', basename) # 尝试匹配 'checkpoint-数字'
                
                current_step_index = -1
                if match:
                    current_step_index = int(match.group(1))
                else:
                    # 如果无法从路径名提取，可以使用列表长度作为备用，但这可能不准确
                    print(f"    Warning: Could not extract step index from {basename}. Using sequential index {len(step_indices)}.")
                    current_step_index = len(step_indices) 

                loaded_loss = torch.load(loss_file)
                if isinstance(loaded_loss, list) or isinstance(loaded_loss, tuple):
                    losses_list.append(torch.tensor(loaded_loss))
                    step_indices.append(current_step_index) # 添加对应的 step_index
                elif isinstance(loaded_loss, torch.Tensor):
                    losses_list.append(loaded_loss)
                    step_indices.append(current_step_index) # 添加对应的 step_index
                else:
                    print(f"    Warning: Could not convert loaded loss from {loss_file} to tensor (type: {type(loaded_loss)}). Skipping.")
                    continue    
            except Exception as e:
                print(f"    Warning: Could not load losses from {loss_file}. Error: {e}")
                continue
        elif os.path.isdir(ckpt_path):
            print(f"  Warning: 'losses.pt' not found in {ckpt_path}")

    if not losses_list:
        print("No losses were successfully loaded.")
        return None, None # 修改：返回两个 None

    # losses.pt 存储的是一个 checkpoint 上所有样本的 loss 值 (1D tensor)
    
    # Check if all tensors in losses_list have the same shape
    # This is crucial before stacking. If they are 1D tensors of losses for all samples at a checkpoint:
    try:
        stacked_losses = torch.stack(losses_list)
    except RuntimeError as e:
        print(f"Error stacking losses. Ensure all loss tensors have compatible shapes. {e}")
        print("Attempting to stack assuming 1D tensors of losses per checkpoint...")
        # Pad if necessary, or ensure all files have the same number of entries.
        return None, None # 修改：返回两个 None

    processed_losses = stacked_losses.t() # Transpose to get (num_samples, num_checkpoints)

    if (num_loss_ckpts > -1) and (processed_losses.shape[1] > num_loss_ckpts) and (num_loss_ckpts > 0):
        print(f"Original number of checkpoints: {processed_losses.shape[1]}")
        # Ensure num_loss_ckpts is not zero to avoid division by zero
        keep_every = processed_losses.shape[1] // num_loss_ckpts
        if keep_every == 0: # handle case where num_loss_ckpts > processed_losses.shape[1]
            keep_every = 1 
            print(f"  Warning: num_loss_ckpts ({num_loss_ckpts}) is greater than available checkpoints ({processed_losses.shape[1]}). Using all available checkpoints.")
        
        print(f"Keeping every {keep_every}-th loss from {processed_losses.shape[1]} checkpoints to get approximately {num_loss_ckpts} checkpoints.")
        indices_to_keep = np.arange(0, processed_losses.shape[1], keep_every)
        if len(indices_to_keep) > num_loss_ckpts and len(indices_to_keep) > 1: # Further trim if we got more than requested
             indices_to_keep = indices_to_keep[:num_loss_ckpts]

        processed_losses = processed_losses[:, indices_to_keep]
        step_indices = [step_indices[i] for i in indices_to_keep] # 对 step_indices 进行下采样
        print(f"Number of checkpoints after subsampling: {processed_losses.shape[1]}")
    else:
        print(f"Using all {processed_losses.shape[1]} loaded loss checkpoints.")

    # Handle NaNs
    processed_losses[torch.isnan(processed_losses)] = 0 # Or some other imputation strategy

    if train_idx is not None:
        print(f"Filtering losses by train_idx. Original shape: {processed_losses.shape}")
        processed_losses = processed_losses[train_idx]
        print(f"Shape after applying train_idx: {processed_losses.shape}")
    
    print(f"Final losses shape: {processed_losses.shape}")
    return processed_losses, step_indices # 修改：返回 losses 和 step_indices

def plot_clustered_loss_trajectories(loss_features, n_clusters, plot_save_path, 
                                     dataset_sample_labels=None, dataset_names=None,
                                     step_indices=None): # 新增 step_indices 参数
    """
    对提供的 loss 特征进行 K-Means 聚类，并绘制每个聚类的 loss 轨迹。
    同时统计每个聚类中来自不同数据集的样本比例。

    参数:
        loss_features (torch.Tensor or np.ndarray): Loss 特征数据，形状为 (num_samples, num_timesteps/checkpoints)。
        n_clusters (int): K-Means 聚类的数量。
        plot_save_path (str): 绘制的图像的保存路径 (例如, './loss_trajectories.png')。
    """
    if isinstance(loss_features, torch.Tensor):
        loss_features_np = loss_features.numpy()
    else:
        loss_features_np = loss_features

    if loss_features_np.shape[0] < n_clusters:
        print(f"Warning: Number of samples ({loss_features_np.shape[0]}) is less than n_clusters ({n_clusters}). Setting n_clusters to {loss_features_np.shape[0]}.")
        n_clusters = loss_features_np.shape[0]
        if n_clusters == 0:
            print("Error: No loss features to plot.")
            return

    print(f"Performing K-Means clustering with {n_clusters} clusters on {loss_features_np.shape[0]} samples...")
    kmeans = faiss.Kmeans(d=loss_features_np.shape[1], k=n_clusters, niter=20, verbose=False, gpu=False) # verbose=True for more logs
    kmeans.train(loss_features_np.astype(np.float32)) # Kmeans expects float32

    # 获取聚类标签
    _, cluster_assignments = kmeans.index.search(loss_features_np.astype(np.float32), 1)
    cluster_assignments = cluster_assignments.flatten()

    print("\nCluster Composition Analysis:")
    for i in range(n_clusters):
        member_indices = np.where(cluster_assignments == i)[0]
        total_in_cluster = len(member_indices)
        if total_in_cluster == 0:
            print(f"Cluster {i}: 0 samples")
            continue
        
        print(f"Cluster {i} (Total: {total_in_cluster} samples):")
        if dataset_sample_labels is not None and dataset_names is not None and len(dataset_sample_labels) == loss_features_np.shape[0]:
            cluster_member_actual_labels = [dataset_sample_labels[idx] for idx in member_indices]
            label_counts = Counter(cluster_member_actual_labels)
            for label_idx, count in sorted(label_counts.items()):
                name = dataset_names[label_idx] if label_idx < len(dataset_names) else f"UnknownDataset_{label_idx}"
                percentage = (count / total_in_cluster) * 100
                print(f"    - {name}: {count} samples ({percentage:.2f}%)")
        else:
            print("    Dataset source information not provided or mismatched.")

    print("\nPlotting loss trajectories by cluster...")
    plt.figure(figsize=(12, 8))
    
    # 使用 colormap 获取颜色
    # colors = plt.cm.get_cmap('viridis', n_clusters) # get_cmap is deprecated
    try:
        colors = plt.colormaps.get_cmap('viridis')(np.linspace(0, 1, n_clusters))
    except AttributeError: # Fallback for older matplotlib
        colors = plt.cm.get_cmap('viridis', n_clusters)
        colors = [colors(i) for i in range(n_clusters)]


    for cluster_id in range(n_clusters):
        member_indices = np.where(cluster_assignments == cluster_id)[0]
        if len(member_indices) == 0:
            print(f"Cluster {cluster_id} has no members.")
            continue
        
        # 绘制该聚类中每个成员的 loss 轨迹
        for member_idx in member_indices:
            loss_trajectory = loss_features_np[member_idx]
            # 使用 step_indices 作为 x 轴数据，如果提供了的话
            current_x_axis = step_indices if step_indices and len(step_indices) == len(loss_trajectory) else np.arange(len(loss_trajectory))
            plt.plot(current_x_axis, loss_trajectory, color=colors[cluster_id], alpha=0.3)
    
    plt.title(f'Loss Trajectories by Cluster (k={n_clusters})')
    plt.xlabel('Checkpoint/Step Index') # 修改横轴标签
    plt.ylabel('Loss')
    
    # 创建图例的代理艺术家
    active_clusters = sorted(list(set(cluster_assignments)))
    if active_clusters:
        # Modify legend to include sample counts if desired, or keep as is
        legend_handles = [plt.Line2D([0], [0], color=colors[i], lw=2) for i in active_clusters]
        # Add sample count to legend label
        legend_labels = []
        for cluster_id_val in active_clusters:
            count = np.sum(cluster_assignments == cluster_id_val)
            legend_labels.append(f'Cluster {cluster_id_val} (n={count})')
        plt.legend(legend_handles, legend_labels, title="Clusters")

    plt.grid(True)
    
    # 确保输出目录存在
    output_dir = os.path.dirname(plot_save_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    plt.savefig(plot_save_path)
    print(f"Saved loss trajectory plot to {plot_save_path}")
    plt.close() # 关闭图形，释放内存

def plot_average_loss_trajectories_by_cluster(loss_features, n_clusters, plot_save_path,
                                              dataset_sample_labels=None, dataset_names=None,
                                              step_indices=None): # 新增 step_indices 参数
    """
    对提供的 loss 特征进行 K-Means 聚类，并绘制每个聚类的平均 loss 轨迹。
    同时统计每个聚类中来自不同数据集的样本比例。

    参数:
        loss_features (torch.Tensor or np.ndarray): Loss 特征数据，形状为 (num_samples, num_timesteps/checkpoints)。
        n_clusters (int): K-Means 聚类的数量。
        plot_save_path (str): 绘制的图像的保存路径 (例如, './average_loss_trajectories.png')。
    """
    if isinstance(loss_features, torch.Tensor):
        loss_features_np = loss_features.numpy()
    else:
        loss_features_np = loss_features

    if loss_features_np.shape[0] < n_clusters:
        print(f"Warning: Number of samples ({loss_features_np.shape[0]}) is less than n_clusters ({n_clusters}). Setting n_clusters to {loss_features_np.shape[0]}.")
        n_clusters = loss_features_np.shape[0]
        if n_clusters == 0:
            print("Error: No loss features to plot.")
            return

    print(f"Performing K-Means clustering with {n_clusters} clusters on {loss_features_np.shape[0]} samples for average plot...")
    kmeans = faiss.Kmeans(d=loss_features_np.shape[1], k=n_clusters, niter=20, verbose=False, gpu=False)
    kmeans.train(loss_features_np.astype(np.float32))

    _, cluster_assignments = kmeans.index.search(loss_features_np.astype(np.float32), 1)
    cluster_assignments = cluster_assignments.flatten()

    print("\nAverage Trajectory Cluster Composition Analysis:")
    cluster_legend_labels_with_composition = []
    for i in range(n_clusters):
        member_indices = np.where(cluster_assignments == i)[0]
        total_in_cluster = len(member_indices)
        if total_in_cluster == 0:
            print(f"Cluster {i}: 0 samples (for average plot)")
            continue

        current_cluster_legend = f'Cluster {i} (avg, n={total_in_cluster}'
        print(f"Cluster {i} (Total: {total_in_cluster} samples for average plot):")
        if dataset_sample_labels is not None and dataset_names is not None and len(dataset_sample_labels) == loss_features_np.shape[0]:
            cluster_member_actual_labels = [dataset_sample_labels[idx] for idx in member_indices]
            label_counts = Counter(cluster_member_actual_labels)
            composition_details = []
            for label_idx, count in sorted(label_counts.items()):
                name = dataset_names[label_idx] if label_idx < len(dataset_names) else f"UnknownDataset_{label_idx}"
                percentage = (count / total_in_cluster) * 100
                print(f"    - {name}: {count} samples ({percentage:.2f}%)")
                composition_details.append(f"{name.split('_')[0][:3]}:{percentage:.0f}%") # Short name for legend
            current_cluster_legend += f" | {', '.join(composition_details)})"
        else:
            print("    Dataset source information not provided or mismatched for average plot.")
            current_cluster_legend += f")"
        cluster_legend_labels_with_composition.append(current_cluster_legend)

    print("\nPlotting average loss trajectories by cluster...")
    plt.figure(figsize=(12, 8))

    try:
        colors = plt.colormaps.get_cmap('viridis')(np.linspace(0, 1, n_clusters))
    except AttributeError: # Fallback for older matplotlib
        cmap = plt.cm.get_cmap('viridis', n_clusters)
        colors = [cmap(i) for i in range(n_clusters)]

    for cluster_id in range(n_clusters):
        member_indices = np.where(cluster_assignments == cluster_id)[0]
        if len(member_indices) == 0:
            # print(f"Cluster {cluster_id} has no members for average plot.") # Already printed above
            continue
        
        average_loss_trajectory = np.mean(loss_features_np[member_indices], axis=0)
        # 使用 step_indices 作为 x 轴数据，如果提供了的话
        current_x_axis = step_indices if step_indices and len(step_indices) == len(average_loss_trajectory) else np.arange(len(average_loss_trajectory))
        
        legend_label_for_plot = f'Cluster {cluster_id}' # Default
        for lbl in cluster_legend_labels_with_composition:
            if lbl.startswith(f'Cluster {cluster_id} ('):
                legend_label_for_plot = lbl
                break
        plt.plot(current_x_axis, average_loss_trajectory, color=colors[cluster_id], linewidth=2.5, label=legend_label_for_plot)
    
    plt.title(f'Average Loss Trajectories by Cluster (k={n_clusters})')
    plt.xlabel('Checkpoint/Step Index') # 修改横轴标签
    plt.ylabel('Average Loss')
    plt.legend(title="Clusters and Composition")
    plt.grid(True)
    
    output_dir = os.path.dirname(plot_save_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    plt.savefig(plot_save_path)
    print(f"Saved average loss trajectory plot to {plot_save_path}")
    plt.close()

# New function for per-dataset processing
def cluster_save_and_plot_per_dataset(
    dataset_name,
    original_data_path, # Path to the original data file for this dataset
    loss_features_for_dataset, # Numpy array of (n_samples_in_this_dataset, n_checkpoints)
    n_clusters,
    base_output_dir,
    plot_individual_trajectories=True, # Whether to plot all trajectories or just average
    skip_plots_if_too_many_samples=True, # For individual trajectories
    max_samples_for_individual_plot=1000, # Threshold
    step_indices=None, # Add step_indices for plotting x-axis
    min_cluster_size_threshold=10 # New parameter for minimum cluster size
):
    """
    Performs K-Means clustering for a single dataset's loss trajectories,
    saves the split data, and plots the results.
    """
    print(f"\n--- Processing dataset: {dataset_name} ---")
    print(f"Original data path: {original_data_path}")
    print(f"Loss features shape for this dataset: {loss_features_for_dataset.shape}")

    output_data_dir = os.path.join(base_output_dir, "split_data", dataset_name)
    output_plot_dir = os.path.join(base_output_dir, "plots", dataset_name)
    os.makedirs(output_data_dir, exist_ok=True)
    os.makedirs(output_plot_dir, exist_ok=True)

    if loss_features_for_dataset.shape[0] == 0:
        print(f"No loss features for dataset {dataset_name}. Skipping.")
        return
    
    # Remove the automatic adjustment of n_clusters based on sample count
    # current_n_clusters = n_clusters
    # if loss_features_for_dataset.shape[0] < current_n_clusters:
    #     print(f"Warning: Number of samples for {dataset_name} ({loss_features_for_dataset.shape[0]}) is less than n_clusters ({current_n_clusters}). Setting n_clusters to {loss_features_for_dataset.shape[0]}.")
    #     current_n_clusters = loss_features_for_dataset.shape[0]
    
    if n_clusters == 0: # Should not happen if previous check passes, but as a safeguard
        print(f"Error: Initial n_clusters is 0. Cannot cluster.")
        return
    if loss_features_for_dataset.shape[1] == 0:
        print(f"Error: Loss features for {dataset_name} have 0 dimensions (checkpoints). Cannot cluster.")
        return

    current_n_clusters = n_clusters
    max_retries = 5 # Limit the number of retries to prevent infinite loops
    retry_count = 0

    while retry_count < max_retries:
        print(f"Performing K-Means clustering with {current_n_clusters} clusters on {loss_features_for_dataset.shape[0]} samples for {dataset_name} (Attempt {retry_count + 1})...")
        
        # Ensure current_n_clusters does not exceed the number of samples
        if current_n_clusters > loss_features_for_dataset.shape[0]:
            print(f"Warning: Adjusted n_clusters ({current_n_clusters}) is greater than number of samples ({loss_features_for_dataset.shape[0]}). Setting n_clusters to {loss_features_for_dataset.shape[0]}.")
            current_n_clusters = loss_features_for_dataset.shape[0]
            if current_n_clusters == 0: # Edge case if dataset has no samples
                print(f"Error: No samples in {dataset_name} to cluster after adjustment (n_clusters became 0).")
                return

        kmeans = faiss.Kmeans(d=loss_features_for_dataset.shape[1], k=current_n_clusters, niter=20, verbose=False, gpu=False)
        kmeans.train(loss_features_for_dataset.astype(np.float32))
        _, cluster_assignments_for_dataset = kmeans.index.search(loss_features_for_dataset.astype(np.float32), 1)
        cluster_assignments_for_dataset = cluster_assignments_for_dataset.flatten()

        # Check cluster sizes
        small_clusters_count = 0
        valid_clusters_count = 0
        for i in range(current_n_clusters):
            member_indices_in_cluster = np.where(cluster_assignments_for_dataset == i)[0]
            if len(member_indices_in_cluster) < min_cluster_size_threshold:
                small_clusters_count += 1
            else:
                valid_clusters_count += 1
        
        print(f"  Found {small_clusters_count} clusters with less than {min_cluster_size_threshold} samples.")
        print(f"  Found {valid_clusters_count} clusters with {min_cluster_size_threshold} or more samples.")

        # If the number of valid clusters is less than the initial n_clusters, increase n_clusters and retry
        if valid_clusters_count < n_clusters and small_clusters_count > 0:
            current_n_clusters += small_clusters_count # Increase n_clusters by the number of small clusters
            print(f"  Increasing n_clusters to {current_n_clusters} and re-clustering.")
            retry_count += 1
        else:
            print(f"  Clustering successful with {valid_clusters_count} valid clusters (>= {min_cluster_size_threshold} samples) and {small_clusters_count} small clusters (< {min_cluster_size_threshold} samples).")
            break # Exit loop if conditions are met
    
    if retry_count == max_retries:
        print(f"Warning: Max retries ({max_retries}) reached for {dataset_name}. Proceeding with current clustering results.")

    # Load original data for splitting
    original_data = []
    if os.path.exists(original_data_path):
        if original_data_path.endswith(".jsonl"):
            original_data = simple_load_jsonl(original_data_path)
        elif original_data_path.endswith(".json"):
            original_data = simple_jload(original_data_path)
        else:
            print(f"Warning: Cannot load original data from {original_data_path} for splitting. Unsupported format.")
    else:
        print(f"Warning: Original data file not found at {original_data_path}. Cannot split data.")

    # #去掉sample数目检查
    # if original_data and len(original_data) != loss_features_for_dataset.shape[0]:
    #     print(f"Warning: Mismatch between number of original data samples ({len(original_data)} from {original_data_path}) and loss features ({loss_features_for_dataset.shape[0]}) for {dataset_name}. Skipping data splitting.")
    #     original_data = [] # Prevent saving incorrect splits

    print(f"\nSplitting and saving data for {dataset_name}:")
    for i in range(current_n_clusters):
        member_indices_in_dataset = np.where(cluster_assignments_for_dataset == i)[0]
        total_in_cluster = len(member_indices_in_dataset)
        print(f"  Cluster {i}: {total_in_cluster} samples")

        if original_data and total_in_cluster > 0:
            # Filter indices to be within the bounds of original_data，去掉为重复样本添加的无效索引
            valid_member_indices = [idx for idx in member_indices_in_dataset if idx < len(original_data)]
            
            if not valid_member_indices:
                print(f"    No valid samples in original_data for cluster {i} after filtering. Skipping save.")
                continue

            cluster_data_samples = [original_data[idx] for idx in valid_member_indices]
            
            if not cluster_data_samples:
                 print(f"    No data samples collected for cluster {i} from {dataset_name} after filtering. Skipping save.")
                 continue

            file_extension = ".jsonl" if original_data_path.endswith(".jsonl") else ".json"
            save_path = os.path.join(output_data_dir, f"cluster_{i}{file_extension}")
            save_data_to_file(cluster_data_samples, save_path)

    # Plotting for this dataset
    # 1. Plot all trajectories colored by cluster
    if plot_individual_trajectories:
        if skip_plots_if_too_many_samples and loss_features_for_dataset.shape[0] > max_samples_for_individual_plot:
            print(f"Skipping individual trajectory plot for {dataset_name} due to large number of samples ({loss_features_for_dataset.shape[0]} > {max_samples_for_individual_plot}).")
        else:
            plot_save_path_individual = os.path.join(output_plot_dir, f"all_trajectories_clustered.png")
            plt.figure(figsize=(12, 8))
            try:
                colors = plt.colormaps.get_cmap('viridis')(np.linspace(0, 1, current_n_clusters))
            except AttributeError: # Fallback for older matplotlib
                cmap_fallback = plt.cm.get_cmap('viridis', current_n_clusters)
                colors = [cmap_fallback(i) for i in range(current_n_clusters)]

            for cluster_id in range(current_n_clusters):
                member_indices = np.where(cluster_assignments_for_dataset == cluster_id)[0]
                if len(member_indices) == 0:
                    continue
                for member_idx in member_indices: # Iterate up to a certain limit if too many, e.g. member_indices[:500]
                    loss_trajectory = loss_features_for_dataset[member_idx]
                    current_x_axis = step_indices if step_indices and len(step_indices) == len(loss_trajectory) else np.arange(len(loss_trajectory))
                    plt.plot(current_x_axis, loss_trajectory, color=colors[cluster_id], alpha=0.3)
            
            plt.title(f'Loss Trajectories by Cluster for {dataset_name} (k={current_n_clusters})')
            plt.xlabel('Checkpoint/Step Index')
            plt.ylabel('Loss')
            active_clusters_ds = sorted(list(set(cluster_assignments_for_dataset)))
            if active_clusters_ds: # Ensure there are active clusters to create a legend for
                legend_handles_ds = [plt.Line2D([0], [0], color=colors[i], lw=2) for i in active_clusters_ds if i < len(colors)]
                legend_labels_ds = [f'Cluster {cid} (n={np.sum(cluster_assignments_for_dataset == cid)})' for cid in active_clusters_ds]
                if legend_handles_ds: # Ensure handles were created
                    plt.legend(legend_handles_ds, legend_labels_ds, title="Clusters")
            plt.grid(True)
            plt.savefig(plot_save_path_individual)
            print(f"Saved individual trajectory plot for {dataset_name} to {plot_save_path_individual}")
            plt.close()

    # 2. Plot average trajectories by cluster
    plot_save_path_avg = os.path.join(output_plot_dir, f"average_trajectories_clustered.png")
    plt.figure(figsize=(12, 8))
    try:
        colors_avg = plt.colormaps.get_cmap('viridis')(np.linspace(0, 1, current_n_clusters))
    except AttributeError: # Fallback for older matplotlib
        cmap_fallback_avg = plt.cm.get_cmap('viridis', current_n_clusters)
        colors_avg = [cmap_fallback_avg(i) for i in range(current_n_clusters)]
    
    legend_labels_avg_list = []
    plotted_anything_avg = False
    for cluster_id in range(current_n_clusters):
        member_indices = np.where(cluster_assignments_for_dataset == cluster_id)[0]
        if len(member_indices) == 0:
            continue
        average_loss_trajectory = np.mean(loss_features_for_dataset[member_indices], axis=0)
        legend_label = f'Cluster {cluster_id} (n={len(member_indices)})'
        current_color = colors_avg[cluster_id % len(colors_avg)]
        current_x_axis_avg = step_indices if step_indices and len(step_indices) == len(average_loss_trajectory) else np.arange(len(average_loss_trajectory))
        plt.plot(current_x_axis_avg, average_loss_trajectory, color=current_color, linewidth=2.5, label=legend_label)
        legend_labels_avg_list.append(legend_label) 
        plotted_anything_avg = True

    plt.title(f'Average Loss Trajectories by Cluster for {dataset_name} (k={current_n_clusters})')
    plt.xlabel('Checkpoint/Step Index')
    plt.ylabel('Average Loss')
    if plotted_anything_avg: 
         plt.legend(title="Clusters", fontsize='small')
    plt.grid(True)
    plt.savefig(plot_save_path_avg)
    print(f"Saved average trajectory plot for {dataset_name} to {plot_save_path_avg}")
    plt.close()
    print(f"--- Finished processing dataset: {dataset_name} ---")


if __name__ == '__main__':
    print("Starting script: plot_loss_clusters...")
    # --- Existing example usage parameters ---
    example_ref_model_path = "./Model/MergeModel/DeepSeek-R1-Distill-Qwen-0.5B-long_short-sft-more_ckpt"
    example_num_loss_ckpts = 17 # -1 to use all, or specify a number
    example_n_clusters = 4

    example_per_dataset_output_base_dir = "./output_per_dataset_analysis"

    example_training_data_paths = [
        "./data/simplescaling/s1K-1.1/data/train.json", 
        "./data/VLyb/s1K-mix/s1_brief_cot.json",
        "./data/N8Programs/gsm8k-r1/train.jsonl",
        "./data/N8Programs/gsm8k-gpt4o/train.json" # Ensure this file exists if uncommented
    ]
    example_training_data_paths = [p for p in example_training_data_paths if os.path.exists(p)]
    if not example_training_data_paths:
        print("Error: No valid training data paths found from 'example_training_data_paths'. Please check the paths.")
        exit()
    print(f"Using training data paths: {example_training_data_paths}")

    example_train_idx = None 

    print("Loading all loss features and step indices...")
    all_loss_features, all_step_indices = load_losses_from_checkpoints(
        ref_model_path=example_ref_model_path,
        num_loss_ckpts=example_num_loss_ckpts,
        train_idx=example_train_idx
    )

    if all_loss_features is None or all_loss_features.shape[0] == 0:
        print("No loss features loaded. Exiting.")
        exit()
    
    print(f"Total loss features loaded shape: {all_loss_features.shape}")
    if all_step_indices:
        print(f"Loaded {len(all_step_indices)} step indices: {all_step_indices[:10]}...")
    else:
        print("No step indices were loaded. Plots will use sequential checkpoint numbers.")

    print("\nLoading dataset source information...")
    _all_sample_source_labels_unfiltered, unique_dataset_names = load_dataset_info_from_paths(example_training_data_paths)

    if not unique_dataset_names:
        print("No dataset names loaded from `load_dataset_info_from_paths`. Cannot proceed with per-dataset analysis. Exiting.")
        exit()

    all_sample_source_labels = np.array(_all_sample_source_labels_unfiltered)
    if example_train_idx is not None:
        print(f"Applying train_idx to dataset source labels. Original count: {len(all_sample_source_labels)}")
        try:
            if isinstance(example_train_idx, torch.Tensor):
                example_train_idx = example_train_idx.numpy()
            
            if np.max(example_train_idx) >= len(all_sample_source_labels) or np.min(example_train_idx) < 0 :
                 print(f"Error: train_idx seems incompatible with the loaded dataset source labels. Max index in train_idx: {np.max(example_train_idx)}, labels_len: {len(all_sample_source_labels)}")
                 exit()
            all_sample_source_labels = all_sample_source_labels[example_train_idx]
            print(f"Count after applying train_idx: {len(all_sample_source_labels)}")
        except Exception as e:
            print(f"Error applying train_idx to source labels: {e}")
            exit()

    if len(all_sample_source_labels) != all_loss_features.shape[0]:
        print(f"Error: Mismatch between number of loss features ({all_loss_features.shape[0]}) and number of sample source labels ({len(all_sample_source_labels)}).")
        print("This can happen if train_idx is applied inconsistently, or if dataset paths/contents do not match the assumptions for loss generation.")
        exit()

    # # --- Optional: Combined analysis (original functionality) ---
    # print("\n--- Starting Combined Analysis (All Samples Together) ---")
    # combined_plot_save_path = os.path.join(example_per_dataset_output_base_dir, "all_samples_clustered_trajectories.png")
    # combined_avg_plot_save_path = os.path.join(example_per_dataset_output_base_dir, "all_samples_average_trajectories.png")
    # plot_clustered_loss_trajectories(
    #     all_loss_features, example_n_clusters, combined_plot_save_path,
    #     dataset_sample_labels=all_sample_source_labels, dataset_names=unique_dataset_names,
    #     step_indices=all_step_indices
    # )
    # plot_average_loss_trajectories_by_cluster(
    #     all_loss_features, example_n_clusters, combined_avg_plot_save_path,
    #     dataset_sample_labels=all_sample_source_labels, dataset_names=unique_dataset_names,
    #     step_indices=all_step_indices
    # )
    # print("--- Finished Combined Analysis ---")

    print("\n--- Starting Per-Dataset Analysis ---")

    for dataset_id, dataset_name in enumerate(unique_dataset_names):
        if dataset_id >= len(example_training_data_paths):
            print(f"Warning: dataset_id {dataset_id} is out of bounds for example_training_data_paths (len: {len(example_training_data_paths)}). Skipping dataset {dataset_name}.")
            continue
        original_data_path_for_current_dataset = example_training_data_paths[dataset_id]
        
        indices_for_current_dataset_in_all_losses = np.where(all_sample_source_labels == dataset_id)[0]

        if len(indices_for_current_dataset_in_all_losses) == 0:
            print(f"No samples found for dataset {dataset_name} (ID: {dataset_id}) in the loaded (and potentially filtered) loss features. Skipping.")
            continue
            
        current_loss_features_for_dataset_tensor = all_loss_features[indices_for_current_dataset_in_all_losses]
        if isinstance(current_loss_features_for_dataset_tensor, torch.Tensor):
            loss_features_for_this_dataset_np = current_loss_features_for_dataset_tensor.numpy()
        elif isinstance(current_loss_features_for_dataset_tensor, np.ndarray):
             loss_features_for_this_dataset_np = current_loss_features_for_dataset_tensor
        else:
            print(f"Error: all_loss_features subset is of unexpected type: {type(current_loss_features_for_dataset_tensor)}. Exiting.")
            exit()

        cluster_save_and_plot_per_dataset(
            dataset_name=dataset_name,
            original_data_path=original_data_path_for_current_dataset,
            loss_features_for_dataset=loss_features_for_this_dataset_np,
            n_clusters=example_n_clusters,
            base_output_dir=example_per_dataset_output_base_dir,
            step_indices=all_step_indices # Pass the global step_indices here
        )
        
    print("\n--- Finished Per-Dataset Analysis ---")
    print("Script finished.")