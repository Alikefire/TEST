import os
import json
import random
from sklearn.model_selection import train_test_split

# Try to import jsonlines, if not available, provide a message
try:
    import jsonlines
except ImportError:
    print("jsonlines library is not installed. Please install it using 'pip install jsonlines' to handle .jsonl files.")
    jsonlines = None

def sample_and_split_data(base_data_path, output_base_path):
    """
    Processes and splits data from sub-directories of base_data_path.

    For each sub-directory (representing a subset like 'math', 'code', 'science'):
    1. Iterates through each .json or .jsonl file (representing a cluster).
    2. For each cluster file, splits its data into 90% training, 5% validation, and 5% evaluation.
    3. Saves the 90% training data for that specific cluster as a separate file.
    4. Merges all validation data from clusters within the same subset into one validation file for that subset.
    5. Merges all evaluation data from clusters within the same subset into one evaluation file for that subset.

    Args:
        base_data_path (str): The path to the directory containing the subset folders.
        output_base_path (str): The path where the processed train, validation, and evaluation
                                files will be saved.
    """
    if not os.path.exists(base_data_path):
        print(f"Error: Base data path '{base_data_path}' does not exist.")
        return

    # Create separate directories for train, validation, and evaluation data
    train_base_path = os.path.join(output_base_path, "train")
    validation_base_path = os.path.join(output_base_path, "validation")
    validation_cluster_base_path = os.path.join(output_base_path, "validation_cluster")
    evaluation_base_path = os.path.join(output_base_path, "evaluation")
    
    os.makedirs(train_base_path, exist_ok=True)
    os.makedirs(validation_base_path, exist_ok=True)
    os.makedirs(validation_cluster_base_path, exist_ok=True)
    os.makedirs(evaluation_base_path, exist_ok=True)

    for subset_name in os.listdir(base_data_path):
        subset_path = os.path.join(base_data_path, subset_name)
        if not os.path.isdir(subset_path):
            continue

        print(f"Processing subset: {subset_name}...")
        
        # Create subset directories in each data type folder
        subset_train_path = os.path.join(train_base_path, subset_name)
        subset_validation_path = os.path.join(validation_base_path, subset_name)
        subset_validation_cluster_path = os.path.join(validation_cluster_base_path, subset_name)
        subset_evaluation_path = os.path.join(evaluation_base_path, subset_name)
        
        os.makedirs(subset_train_path, exist_ok=True)
        os.makedirs(subset_validation_path, exist_ok=True)
        os.makedirs(subset_validation_cluster_path, exist_ok=True)
        os.makedirs(subset_evaluation_path, exist_ok=True)
        

        subset_all_validation_data = []
        subset_all_evaluation_data = []

        for filename in os.listdir(subset_path):
            file_path = os.path.join(subset_path, filename)
            cluster_data = []
            original_filename_no_ext, _ = os.path.splitext(filename)

            if filename.endswith('.json'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            cluster_data.extend(data)
                        else:
                            print(f"Warning: JSON file {file_path} does not contain a list. Skipping.")
                            continue
                except json.JSONDecodeError:
                    print(f"Error decoding JSON from {file_path}. Skipping.")
                    continue
                except Exception as e:
                    print(f"An unexpected error occurred while reading {file_path}: {e}. Skipping.")
                    continue
            elif filename.endswith('.jsonl'):
                if jsonlines:
                    try:
                        with jsonlines.open(file_path, 'r') as reader:
                            for item in reader:
                                cluster_data.append(item)
                    except Exception as e:
                        print(f"An unexpected error occurred while reading {file_path}: {e}. Skipping.")
                        continue
                else:
                    print(f"Skipping .jsonl file {file_path} as jsonlines library is not available.")
                    continue
            else:
                continue

            if not cluster_data:
                print(f"No data found in cluster file: {filename} for subset: {subset_name}. Skipping.")
                continue
            
            print(f"  Processing cluster file: {filename} ({len(cluster_data)} items)")
            random.shuffle(cluster_data)

            num_total_items = len(cluster_data)
            num_val_items = max(1, int(0.05 * num_total_items)) if num_total_items > 0 else 0
            num_eval_items = max(1, int(0.05 * num_total_items)) if num_total_items > 0 else 0

            train_data_cluster = []
            val_data_cluster = []
            eval_data_cluster = []

            if num_total_items == 0:
                print(f"  Cluster file {filename} is empty.")
            elif num_total_items < 3 : # Not enough for 1 train, 1 val, 1 eval
                print(f"  Warning: Cluster file {filename} has {num_total_items} items. Assigning to train/val/eval based on availability.")
                if num_total_items == 1:
                    train_data_cluster = cluster_data
                elif num_total_items == 2:
                    train_data_cluster = [cluster_data[0]]
                    val_data_cluster = [cluster_data[1]]
            else:
                # Split: train_data_cluster gets the rest after val and eval are taken
                # temp_data is what's left after taking val_items
                temp_data, val_data_cluster = train_test_split(cluster_data, test_size=num_val_items, shuffle=False)
                # train_data_cluster is what's left after taking eval_items from temp_data
                if len(temp_data) > 0:
                    # Calculate eval_size relative to temp_data, ensuring it doesn't exceed temp_data length
                    eval_size_for_split = min(num_eval_items, len(temp_data))
                    if eval_size_for_split == len(temp_data): # If eval takes all remaining
                         eval_data_cluster = temp_data
                         train_data_cluster = []
                    elif eval_size_for_split > 0:
                        train_data_cluster, eval_data_cluster = train_test_split(temp_data, test_size=eval_size_for_split, shuffle=False)
                    else: # eval_size_for_split is 0
                        train_data_cluster = temp_data
                        eval_data_cluster = []
                else: # temp_data is empty, means all went to val_data_cluster
                    train_data_cluster = []
                    eval_data_cluster = []
            
            # Save training data for the current cluster in train and validation directory 
            if train_data_cluster:
                cluster_train_output_filename = f"{original_filename_no_ext}_train.json"
                cluster_validation_output_filename = f"{original_filename_no_ext}_validation.json"
                cluster_train_output_path = os.path.join(subset_train_path, cluster_train_output_filename)
                cluster_validation_output_path = os.path.join(subset_validation_cluster_path, cluster_validation_output_filename)
                try:
                    with open(cluster_train_output_path, 'w', encoding='utf-8') as f:
                        json.dump(train_data_cluster, f, ensure_ascii=False, indent=4)
                    with open(cluster_validation_output_path, 'w', encoding='utf-8') as f:
                        json.dump(val_data_cluster, f, ensure_ascii=False, indent=4)
                    print(f"    Saved cluster training data to {cluster_train_output_path} ({len(train_data_cluster)} items)")
                except IOError as e:
                    print(f"    Error saving training data for cluster {filename}: {e}")
            else:
                print(f"    No training data to save for cluster {filename}.")

            subset_all_validation_data.extend(val_data_cluster)
            subset_all_evaluation_data.extend(eval_data_cluster)
            print(f"    Cluster {filename} split: Train={len(train_data_cluster)}, Val={len(val_data_cluster)}, Eval={len(eval_data_cluster)}")

        # Save aggregated validation data in validation directory
        if subset_all_validation_data:
            subset_validation_output_path = os.path.join(subset_validation_path, f"{subset_name}_validation.json")
            try:
                with open(subset_validation_output_path, 'w', encoding='utf-8') as f:
                    json.dump(subset_all_validation_data, f, ensure_ascii=False, indent=4)
                print(f"  Saved {subset_name} validation data to {subset_validation_output_path} ({len(subset_all_validation_data)} items)")
            except IOError as e:
                print(f"Error saving validation data for {subset_name}: {e}")
        else:
            print(f"No validation data to save for subset {subset_name}.")

        # Save aggregated evaluation data in evaluation directory
        if subset_all_evaluation_data:
            subset_evaluation_output_path = os.path.join(subset_evaluation_path, f"{subset_name}_evaluation.json")
            try:
                with open(subset_evaluation_output_path, 'w', encoding='utf-8') as f:
                    json.dump(subset_all_evaluation_data, f, ensure_ascii=False, indent=4)
                print(f"  Saved {subset_name} evaluation data to {subset_evaluation_output_path} ({len(subset_all_evaluation_data)} items)")
            except IOError as e:
                print(f"Error saving evaluation data for {subset_name}: {e}")
        else:
            print(f"No evaluation data to save for subset {subset_name}.")

if __name__ == '__main__':
    # IMPORTANT: Replace these paths with your actual directory paths
    input_split_data_directory = './data/open-r1/Mixture-of-Thoughts/mix_train_data/instruct'
    # This will create separate train, validation, and evaluation directories
    output_directory = './data/open-r1/Mixture-of-Thoughts/mix_train_data/instruct_processed'
    
    sample_and_split_data(input_split_data_directory, output_directory)
    print("Data sampling and splitting process finished.")