import torch
import numpy as np
import faiss
import matplotlib.pyplot as plt
import os
import glob
import json # Added for jload/load_jsonl simple implementation
from collections import Counter # Added for counting labels

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
    current_label_idx = 0

    if not isinstance(data_paths_list, list):
        print(f"Warning: data_paths_list is not a list: {data_paths_list}. Treating as single item list.")
        data_paths_list = [data_paths_list]

    for path in data_paths_list:
        dataset_name = os.path.basename(path).split('.')[0] # Simple name extraction
        if dataset_name not in unique_dataset_names:
            unique_dataset_names.append(dataset_name)
        
        num_samples = 0
        try:
            if isinstance(path, str) and path.endswith(".jsonl"):
                data = simple_load_jsonl(path)
                num_samples = len(data)
            elif isinstance(path, str) and path.endswith(".json"):
                data = simple_jload(path)
                num_samples = len(data)
            elif isinstance(path, str): # Assuming Hugging Face dataset identifier
                # This part relies on the 'datasets' library
                hf_dataset = load_dataset(path, split='train') # Or appropriate split
                num_samples = len(hf_dataset)
            else:
                print(f"Warning: Unsupported data path format for label generation: {path}")
        except Exception as e:
            print(f"Error loading data from {path} for label generation: {e}. Skipping this source.")
            continue
        
        all_sample_labels.extend([unique_dataset_names.index(dataset_name)] * num_samples)
        print(f"  Loaded {num_samples} samples from {dataset_name} (label: {unique_dataset_names.index(dataset_name)})")

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
    checkpoint_paths = sorted(glob.glob(os.path.join(ref_model_path, '*')))

    if not checkpoint_paths:
        print(f"No checkpoint directories found in {ref_model_path}")
        return None

    for ckpt_path in checkpoint_paths:
        loss_file = os.path.join(ckpt_path, "losses.pt")
        if os.path.isdir(ckpt_path) and os.path.exists(loss_file):
            print(f"  Loading losses from: {loss_file}")
            try:
                # losses_list.append(torch.tensor(torch.load(loss_file)))
                # Ensure loaded losses are tensors. If they are lists/tuples, convert them.
                loaded_loss = torch.load(loss_file)
                if isinstance(loaded_loss, list) or isinstance(loaded_loss, tuple):
                    losses_list.append(torch.tensor(loaded_loss))
                elif isinstance(loaded_loss, torch.Tensor):
                    losses_list.append(loaded_loss)
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
        return None

    # 确保所有 loss 张量可以被堆叠 (例如，具有相同的样本数量)
    # 如果 loss 张量是一维的 (每个 checkpoint 一个 loss 文件，每个文件包含所有样本的 loss)
    # 那么我们应该堆叠它们，然后转置
    # 如果 loss 张量已经是 (num_samples, 1) 形状，我们应该连接它们
    # 假设 losses.pt 存储的是一个 checkpoint 上所有样本的 loss 值 (1D tensor)
    
    # Check if all tensors in losses_list have the same shape
    # This is crucial before stacking. If they are 1D tensors of losses for all samples at a checkpoint:
    try:
        # Attempt to stack. This assumes each file contains a 1D tensor of losses for all samples.
        # Or, if each file contains a (num_samples, 1) tensor, hstack might be more appropriate after squeeze.
        # For now, let's assume each losses.pt is a 1D tensor [loss_sample1, loss_sample2, ...]
        stacked_losses = torch.stack(losses_list)
    except RuntimeError as e:
        print(f"Error stacking losses. Ensure all loss tensors have compatible shapes. {e}")
        # Try to handle cases where losses might be single float values per file (unlikely for trajectories)
        # Or if they are already (N,1) and need to be concatenated along dim=1
        # For simplicity, if stacking fails, we print an error and return.
        # A more robust solution would inspect shapes and try different stacking/concatenation strategies.
        # Example: if all are (N,1), then torch.cat(losses_list, dim=1)
        # Example: if all are (1,N), then torch.cat(losses_list, dim=0).t()
        # Given the original code's .t() later, it's likely each file is a 1D array of losses for samples.
        print("Attempting to stack assuming 1D tensors of losses per checkpoint...")
        # Pad if necessary, or ensure all files have the same number of entries.
        # This part needs to be robust based on the actual structure of losses.pt
        # For now, we'll proceed with the assumption that stack() should work or it's an error.
        return None

    processed_losses = stacked_losses.t() # Transpose to get (num_samples, num_checkpoints)

    if (num_loss_ckpts > -1) and (processed_losses.shape[1] > num_loss_ckpts):
        print(f"Original number of checkpoints: {processed_losses.shape[1]}")
        keep_every = processed_losses.shape[1] // num_loss_ckpts
        print(f"Keeping every {keep_every}-th loss from {processed_losses.shape[1]} checkpoints to get {num_loss_ckpts} checkpoints.")
        indices_to_keep = np.arange(0, processed_losses.shape[1], keep_every)
        processed_losses = processed_losses[:, indices_to_keep]
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
    return processed_losses

def plot_clustered_loss_trajectories(loss_features, n_clusters, plot_save_path, 
                                     dataset_sample_labels=None, dataset_names=None):
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
            plt.plot(loss_trajectory, color=colors[cluster_id], alpha=0.3) # 使用带透明度的颜色
    
    plt.title(f'Loss Trajectories by Cluster (k={n_clusters})')
    plt.xlabel('Checkpoint/Epoch Index')
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
                                              dataset_sample_labels=None, dataset_names=None):
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
        # Use the detailed legend label generated above
        legend_label_for_plot = f'Cluster {cluster_id}' # Default
        for lbl in cluster_legend_labels_with_composition:
            if lbl.startswith(f'Cluster {cluster_id} ('):
                legend_label_for_plot = lbl
                break
        plt.plot(average_loss_trajectory, color=colors[cluster_id], linewidth=2.5, label=legend_label_for_plot)
    
    plt.title(f'Average Loss Trajectories by Cluster (k={n_clusters})')
    plt.xlabel('Checkpoint/Epoch Index')
    plt.ylabel('Average Loss')
    plt.legend(title="Clusters", fontsize='small') # Adjust legend display if needed
    plt.grid(True)
    
    output_dir = os.path.dirname(plot_save_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    plt.savefig(plot_save_path)
    print(f"Saved average loss trajectory plot to {plot_save_path}")
    plt.close()

if __name__ == '__main__':
    print("Starting example usage of plot_loss_clusters...")
    # --- 示例用法 ---
    
    example_ref_model_path = "./Model/MergeModel/DeepSeek-R1-Distill-Qwen-0.5B-long_short-sft-more_ckpt" 
    example_num_loss_ckpts = 17  
    example_n_clusters = 4       
    example_plot_save_path = "./output_plots/clustered_loss_trajectories.png" 
    example_avg_plot_save_path = "./output_plots/average_loss_trajectories.png"

    # Define the paths to the datasets used for training, in the same order they were loaded
    # IMPORTANT: These paths must correspond to the data that generated the losses.
    # The order and content must match the training run.
    example_training_data_paths = [
        "./data/simplescaling/s1K-1.1/data/train.json", 
        "./data/VLyb/s1K-mix/s1_brief_cot.json",
        "./data/N8Programs/gsm8k-r1/train.jsonl",
        "./data/N8Programs/gsm8k-gpt4o/train.json" # Replace with actual path
        # Add more paths if multiple datasets were concatenated during training
    ]
    # If your SupervisedDataset was initialized with a single path that internally handled multiple files,
    # or if it was a Hugging Face dataset name that combines multiple sources, adjust accordingly.
    # For this example, we assume a list of distinct dataset files/sources.

    # 1. Load dataset source information
    print("\nLoading dataset source information...")
    dataset_sample_labels, unique_dataset_names = load_dataset_info_from_paths(example_training_data_paths)
    if not dataset_sample_labels:
        print("Could not load dataset labels. Proceeding without dataset composition analysis.")
        dataset_sample_labels = None # Ensure it's None if loading failed
        unique_dataset_names = None

    # 2. 加载 loss 数据
    loss_data = load_losses_from_checkpoints(example_ref_model_path, num_loss_ckpts=example_num_loss_ckpts)

    if loss_data is not None and loss_data.nelement() > 0:
        if dataset_sample_labels is not None and len(dataset_sample_labels) != loss_data.shape[0]:
            print(f"Warning: Mismatch between number of loss samples ({loss_data.shape[0]}) and dataset labels ({len(dataset_sample_labels)}).")
            print("Dataset composition analysis might be incorrect. Ensure data_paths for labels match training data.")
            # Decide how to handle: proceed with caution, or disable composition analysis
            # For now, we'll pass them, but the functions will also check length.

        # 3. 进行聚类和绘图 (所有轨迹)
        plot_clustered_loss_trajectories(loss_data, example_n_clusters, example_plot_save_path, 
                                         dataset_sample_labels=dataset_sample_labels, dataset_names=unique_dataset_names)
        print(f"Example finished. Plot saved to {example_plot_save_path if os.path.exists(example_plot_save_path) else 'due to an error'}")
        
        # 4. 进行聚类和绘图 (平均轨迹)
        plot_average_loss_trajectories_by_cluster(loss_data, example_n_clusters, example_avg_plot_save_path,
                                                  dataset_sample_labels=dataset_sample_labels, dataset_names=unique_dataset_names)
        print(f"Example for average plot finished. Plot saved to {example_avg_plot_save_path if os.path.exists(example_avg_plot_save_path) else 'due to an error'}")
    elif loss_data is None:
        print("Failed to load loss data. Skipping plotting.")
    else: # loss_data.nelement() == 0
        print("Loaded loss data is empty. Skipping plotting.")

    # # --- 另一个示例：使用随机生成的 loss 数据 ---
    # print("\nStarting example with randomly generated loss data...")
    # num_samples_rand = 100
    # num_checkpoints_rand = 20
    # random_losses = torch.rand(num_samples_rand, num_checkpoints_rand) * 5 + torch.sin(torch.linspace(0, 10, num_checkpoints_rand)).unsqueeze(0) # 创建一些趋势
    # rand_n_clusters = 4
    # rand_plot_save_path = "./output_plots/random_loss_trajectories.png"
    # rand_avg_plot_save_path = "./output_plots/random_average_loss_trajectories.png"

    # plot_clustered_loss_trajectories(random_losses, rand_n_clusters, rand_plot_save_path,
    #                                  dataset_sample_labels=rand_dataset_labels, dataset_names=rand_dataset_names)
    # print(f"Random data example finished. Plot saved to {rand_plot_save_path if os.path.exists(rand_plot_save_path) else 'due to an error'}")

    # plot_average_loss_trajectories_by_cluster(random_losses, rand_n_clusters, rand_avg_plot_save_path,
    #                                           dataset_sample_labels=rand_dataset_labels, dataset_names=rand_dataset_names)
    # print(f"Random data average plot example finished. Plot saved to {rand_avg_plot_save_path if os.path.exists(rand_avg_plot_save_path) else 'due to an error'}")