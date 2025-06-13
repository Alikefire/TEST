import logging
from dataclasses import dataclass
from datasets import load_dataset
import os
from typing import Union, Dict, Sequence
import io
import copy
import json
import torch
from torch.utils.data import Dataset
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM

from consts import *


## ALPACA-STYLE PROMPT: forked from https://github.com/tatsu-lab/stanford_alpaca
class Prompter(object):
    __slots__ = ("template", "_verbose")
    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        file_name = os.path.join("templates", f"{template_name}.json")
        if not os.path.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()

def tokenize(tokenizer, cutoff_len, prompt, add_eos_token=True):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < cutoff_len
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)
    result["labels"] = result["input_ids"].copy()  # labels = input_ids -> training decoder
    return result

def generate_and_tokenize_prompt(tokenizer, cutoff_len, prompter, train_on_inputs, add_eos_token, data_point):
    full_prompt = prompter.generate_prompt(
        instruction=data_point["instruction"],
        input=data_point["input"],
        label=data_point["output"],
    )
    tokenized_full_prompt = tokenize(tokenizer=tokenizer,
                                     cutoff_len=cutoff_len,
                                     prompt=full_prompt,
                                     add_eos_token=True)  # default
    if not train_on_inputs:
        user_prompt = prompter.generate_prompt(
            data_point["instruction"], data_point["input"]
        )
        tokenized_user_prompt = tokenize(tokenizer=tokenizer,
                                        cutoff_len=cutoff_len,
                                        prompt=user_prompt,
                                        add_eos_token=True
                                        )
        user_prompt_len = len(tokenized_user_prompt["input_ids"])
        if add_eos_token:
            user_prompt_len -= 1
        tokenized_full_prompt["labels"] = [
            -100
        ] * user_prompt_len + tokenized_full_prompt["labels"][
            user_prompt_len:
        ]  # could be sped up, probably
    return tokenized_full_prompt

def get_prompter(prompt_template_name):
    prompter = Prompter(prompt_template_name)
    return prompter


## GET & FIX TOKENIZERS
def get_tokenizer(model_name_or_path, cache_dir, model_max_length, ):
    tokenizer = AutoTokenizer.from_pretrained(
                    model_name_or_path,
                    cache_dir=cache_dir,
                    model_max_length=model_max_length,
                    padding_side="right",
                )
    special_tokens_dict = dict()
    #移出硬编码的special_token
    # special_tokens_dict["pad_token"] = LLAMA_DEFAULT_PAD_TOKEN
    # special_tokens_dict["eos_token"] = LLAMA_DEFAULT_EOS_TOKEN
    # special_tokens_dict["bos_token"] = LLAMA_DEFAULT_BOS_TOKEN
    # special_tokens_dict["unk_token"] = LLAMA_DEFAULT_UNK_TOKEN
    # PROBLEM !!! -> fixed in smart_tokenizer_and_embedding_resize
    # if tokenizer.pad_token is None:
    #     special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    # if tokenizer.eos_token is None:
    #     special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    # if tokenizer.bos_token is None:
    #     special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    # if tokenizer.unk_token is None:
    #     special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
    # FIX --> bos/eos/unk/pad
    # special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    # special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    # special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    # special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
    return tokenizer, special_tokens_dict

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,  
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)  # fix tokenizer special tokens map
    if model!=None:
        model.resize_token_embeddings(len(tokenizer))
        if num_new_tokens > 0:
            input_embeddings = model.get_input_embeddings().weight.data
            output_embeddings = model.get_output_embeddings().weight.data
            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg
    return tokenizer, model

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = LLAMA_IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


## DATASETS / DATALOADER
class SupervisedDataset(Dataset):
    """Dataset for sft."""
    def __init__(self, data_path: Union[str, list[str]], tokenizer: transformers.PreTrainedTokenizer, data_format: Union[str, list[str]] = "MathInstruct"):
        super(SupervisedDataset, self).__init__()
        logging.warning(f"Loading data with format: {data_format}...")
        
        all_list_data_dict = []

        if isinstance(data_path, str):
            # Handle single data_path string
            data_paths_list = [data_path]
        elif isinstance(data_path, list):
            # Handle list of data_path strings
            data_paths_list = data_path
        else:
            raise ValueError(f"data_path must be a string or a list of strings, got {type(data_path)}")

        # Determine data formats for each path
        data_formats_list = []
        if isinstance(data_format, str):
            data_formats_list = [data_format] * len(data_paths_list)
            logging.warning(f"Using single data format '{data_format}' for all {len(data_paths_list)} data paths.")
        elif isinstance(data_format, list):
            if len(data_format) == len(data_paths_list):
                data_formats_list = data_format
                logging.warning(f"Using specific data formats for each data path: {data_formats_list}")
            else:
                raise ValueError(
                    f"Length of data_format list ({len(data_format)}) "
                    f"must match length of data_path list ({len(data_paths_list)})."
                )
        else:
            raise ValueError(f"data_format must be a string or a list of strings, got {type(data_format)}")

        first_data_path_for_prompt = data_paths_list[0] if data_paths_list else ""

        for i, current_path in enumerate(data_paths_list):
            current_file_format = data_formats_list[i]
            logging.warning(f"Processing data from: {current_path} with format: {current_file_format}")
            list_data_dict_single_source = []
            
            # Infer format from path if the specified format is a generic one or needs override
            # This allows for a base format with path-based overrides if desired.
            if current_file_format == "auto" or "sharegpt" in current_path.lower() and current_file_format != "sharegpt":
                if "sharegpt" in current_path.lower():
                    current_file_format = "sharegpt"
                    logging.info(f"  Inferred format as 'sharegpt' for {current_path}")
            elif current_file_format == "auto" or "alpaca" in current_path.lower() and current_file_format != "alpaca":
                if "alpaca" in current_path.lower():
                    current_file_format = "alpaca"
                    logging.info(f"  Inferred format as 'alpaca' for {current_path}")
            # Add more specific inferences if necessary, or if current_file_format is 'auto'

            if current_file_format == "MathInstruct":
                if current_path.endswith(".jsonl"):
                    raw_data = load_jsonl(current_path)
                    for item in raw_data:
                        # Assuming MathInstruct format has 'instruction', 'output', 'source'
                        list_data_dict_single_source.append({
                            'instruction': item.get('instruction', ''),
                            'input': item.get('input', ''), # MathInstruct might use 'input' or not
                            'output': item.get('output', ''),
                            'source': item.get('source', '')
                        })
                elif current_path.endswith(".json"):
                    raw_data = jload(current_path)
                    for item in raw_data:
                        list_data_dict_single_source.append({
                            'instruction': item.get('instruction', ''),
                            'input': item.get('input', ''),
                            'output': item.get('output', ''),
                            'source': item.get('source', '')
                        })
                else:
                    # Fallback for Hugging Face dataset loading if path is not json/jsonl
                    # This part might need adjustment based on how MathInstruct is stored in HF datasets
                    try:
                        loaded_hf_dataset = load_dataset(current_path)["train"]
                        for item in loaded_hf_dataset:
                             list_data_dict_single_source.append({
                                'instruction': item.get('instruction', ''),
                                'input': item.get('input', ''),
                                'output': item.get('output', ''),
                                'source': item.get('source', '')
                            })
                    except Exception as e:
                        logging.error(f"Failed to load MathInstruct from {current_path} as HF dataset: {e}")
                        continue

            elif current_file_format == "sharegpt":
                # Handles .json or .jsonl for sharegpt
                raw_data = []
                if current_path.endswith(".jsonl"):
                    raw_data = load_jsonl(current_path)
                elif current_path.endswith(".json"):
                    raw_data = jload(current_path)
                else:
                    logging.warning(f"ShareGPT format expects .json or .jsonl, got {current_path}")
                    continue
                
                for item in raw_data:
                    if "conversations" in item and isinstance(item["conversations"], list) and len(item["conversations"]) >= 2:
                        human_msg = ""
                        gpt_msg = ""
                        # Iterate through conversations to find human and gpt turns
                        # This simple version takes the first human and subsequent gpt message
                        # More complex logic might be needed for multi-turn conversations
                        for i, conv in enumerate(item["conversations"]):
                            if conv.get("from") == "human":
                                human_msg = conv.get("value", "")
                                # Look for the next gpt message
                                if i + 1 < len(item["conversations"]) and item["conversations"][i+1].get("from") == "gpt":
                                    gpt_msg = item["conversations"][i+1].get("value", "")
                                    break # Found a pair
                        if human_msg and gpt_msg:
                            list_data_dict_single_source.append({
                                'instruction': human_msg, 
                                'input': item.get('system', ''), # Use system prompt as input if available,添加system到input
                                'output': gpt_msg
                            })
                        else:
                            logging.warning(f"Could not parse human/gpt pair from sharegpt item: {item}") 
                    else:
                        logging.warning(f"Skipping malformed sharegpt item: {item}")
            
            elif current_file_format == "alpaca":
                raw_data = []
                if current_path.endswith(".jsonl"):
                    raw_data = load_jsonl(current_path)
                elif current_path.endswith(".json"):
                    raw_data = jload(current_path)
                else:
                    logging.warning(f"Alpaca format expects .json or .jsonl, got {current_path}")
                    continue

                for item in raw_data:
                    # Alpaca format: instruction, input (optional), output. System can be part of instruction or input.
                    instruction = item.get('instruction', '')
                    _input = item.get('input', '')
                    output = item.get('output', '')
                    system_prompt = item.get('system', '')
                    
                    # Combine system prompt with instruction if system prompt exists
                    if system_prompt:
                        # Decide how to combine: prepend to instruction or use as input
                        # Here, prepending to instruction for simplicity
                        instruction = f"{system_prompt}\n\n{instruction}" 

                    list_data_dict_single_source.append({
                        'instruction': instruction,
                        'input': _input, 
                        'output': output
                    })
            
            # Generic loading for other Hugging Face datasets if format not matched
            # This was part of your original code, kept for fallback.
            elif 'Asclepius' in current_path: # Example of specific HF dataset handling
                loaded_hf_dataset = load_dataset(current_path)["train"]
                list_data_dict_single_source = [{'instruction':data['question'], 'input':data.get('note',''), 'output':data['answer'], 'source':data.get('task','')} for data in loaded_hf_dataset]
            else:
                try:
                    logging.warning(f"Attempting generic Hugging Face dataset load for {current_path} as format '{current_file_format}' was not specifically handled or inferred.")
                    loaded_hf_dataset = load_dataset(current_path)["train"]
                    # Adapt this generic loading based on common structures or specific needs
                    for i_hf, hf_item in enumerate(loaded_hf_dataset):
                        if 'conversations' in hf_item and len(hf_item['conversations']) >= 2: # A common pattern
                            list_data_dict_single_source.append(dict(instruction=hf_item['conversations'][0]['value'], output=hf_item['conversations'][1]['value']))
                        elif 'instruction' in hf_item and 'output' in hf_item: # Another common pattern
                             list_data_dict_single_source.append(dict(instruction=hf_item['instruction'], input=hf_item.get('input',''), output=hf_item['output']))
                        else:
                            logging.warning(f"Skipping malformed entry in generic HF load for {current_path} at index {i_hf}: {hf_item}")
                except Exception as e:
                    logging.error(f"Failed to load generic Hugging Face dataset {current_path}: {e}")
                    continue
            
            all_list_data_dict.extend(list_data_dict_single_source)

        if not all_list_data_dict:
            logging.error("No data loaded. Please check data_path, data_format, and file contents.")
            self.input_ids = []
            self.labels = []
            return
            
        logging.warning("Formatting inputs...")
        # Use the first data_path to determine prompt type, or a more sophisticated logic if needed
        if 'mimic' in first_data_path_for_prompt: # Check based on the first path, or make this configurable
            prompt_input, prompt_no_input = RADIOLOGY_PROMPT_DICT["prompt_no_input"], RADIOLOGY_PROMPT_DICT["prompt_no_input"]
        else:
            prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        
        sources = []
        targets = []
        for example in all_list_data_dict:
            # Ensure example is a dict and has 'instruction' and 'output'
            if not isinstance(example, dict):
                logging.warning(f"Skipping non-dict example: {type(example)}")
                continue
            
            instruction = example.get('instruction')
            output = example.get('output')
            _input = example.get('input', "") # Ensure 'input' key exists or default to empty string

            if instruction is None or output is None:
                logging.warning(f"Skipping example with missing 'instruction' or 'output': {example}")
                continue

            # Ensure all parts are strings before formatting
            current_source = prompt_input.format_map({'instruction': str(instruction), 'input': str(_input)}) \
                if _input != "" else prompt_no_input.format_map({'instruction': str(instruction)})
            sources.append(current_source)
            targets.append(f"{str(output)}{tokenizer.eos_token}")

        if not sources or not targets:
            logging.error("No valid sources or targets after formatting. Check data integrity and prompt definitions.")
            self.input_ids = []
            self.labels = []
            return

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

        if len(self.input_ids) != len(all_list_data_dict):
            logging.warning(f"Mismatch in length after tokenization. Initial: {len(all_list_data_dict)}, Tokenized: {len(self.input_ids)}. This might be due to filtering in preprocess or empty/invalid examples.")
        # 添加原始索引列表
        self.original_indices = list(range(len(self.input_ids)))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, Union[torch.Tensor, int]]: # 或者 Dict[str, Any]
        return dict(input_ids=self.input_ids[i], labels=self.labels[i], original_index=self.original_indices[i])

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for sft."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=LLAMA_IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_path, data_format: Union[str, list[str]] = "MathInstruct") -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_path, data_format=data_format)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


## GET LLAMA-MODEL
def get_model(model_name_or_path, cache_dir=None):
    if "t5" in model_name_or_path:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, cache_dir=cache_dir)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, cache_dir=cache_dir)
    return model


## WHITENINING
def compute_kernel_bias(batch_hidden_states):
    """for final transformation: y = (x + bias).dot(kernel)
    batched_hidden_states .shape = (batch_size, hidden_dim)
    """
    mu = batch_hidden_states.mean(axis=0, keepdims=True)  # (1, hidden_dim)
    cov = torch.cov(batch_hidden_states.t())  # (hidden_dim, hidden_dim)
    u, s, vh = torch.linalg.svd(cov)  # u.shape = (hidden_dim, hidden_dim)  s.shape = (hidden_dim)  vh.shape = (hidden_dim, hidden_dim)
    W = torch.mm(u, torch.diag(1/torch.sqrt(s)))  # (hidden_dim, hidden_dim)
    # kernel = W  # (hidden_dim, hidden_dim)
    # bias = -mu  # (batch_size, hidden_dim)
    return W, -mu

def normalize(batch_hidden_states):
    return batch_hidden_states / (batch_hidden_states**2).sum(dim=1, keepdims=True)**0.5

def transform_and_normalize(batch_hidden_states, kernel, bias):
    """apply transformation & normalization
    batched_hidden_states .shape = (batch_size, hidden_dim)
    kernel .shape = (hidden_dim, hidden_dim) --> 取N_COMPONENTS后 (emb_dim, n_dim)
    bias .shape = (batch_size, hidden_dim) 
    """
    if not (kernel is None or bias is None):
        transformed_batch_hidden_states = torch.mm((batch_hidden_states + bias), kernel)  # (batch_size, n_dim)
    return normalize(transformed_batch_hidden_states)  # (batch_size, n_dim)


## JSON - LOAD/DUMP: forked from https://github.com/tatsu-lab/stanford_alpaca
def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.
    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default, ensure_ascii=False)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict



## OTHERS
def rank0_print(message):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank()==0:
            print(message)
        else:
            return
    else:
        print(message)
            

def load_jsonl(file):
    lines = []
    with open(file, "r") as f:
        for line in f.readlines():
            lines.append(json.loads(line))
    return lines

