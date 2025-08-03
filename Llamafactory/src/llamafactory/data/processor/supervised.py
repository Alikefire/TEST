# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, List, Tuple # Ensure List and Tuple are imported

import random  # 新增导入
import copy   # 新增导入

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from .processor_utils import DatasetProcessor, greedy_knapsack, infer_seqlen,random_knapsack


if TYPE_CHECKING:
    from ..mm_plugin import AudioInput, ImageInput, VideoInput
    from ...hparams import DataArguments # 假设 DataArguments 在这里可以被引用


logger = logging.get_logger(__name__)




@dataclass
class SupervisedDatasetProcessor(DatasetProcessor):
    def _encode_data_example(
        self,
        prompt: list[dict[str, str]],
        response: list[dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        current_index: Optional[int] = None, # 新增 current_index 参数
    ) -> tuple[list[int], list[int]]:
        # if current_index is not None: # 日志记录当前处理的样本原始索引
        #     logger.info(f"Encoding data example with original_index: {current_index}")

        messages = self.template.mm_plugin.process_messages(prompt + response, images, videos, audios, self.processor)
        input_ids, labels = self.template.mm_plugin.process_token_ids(
            [], [], images, videos, audios, self.tokenizer, self.processor
        )
        encoded_pairs = self.template.encode_multiturn(self.tokenizer, messages, system, tools)
        total_length = len(input_ids) + (1 if self.template.efficient_eos else 0)
        if self.data_args.mask_history:
            encoded_pairs = encoded_pairs[::-1]  # high priority for last turns

        for turn_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
            if total_length >= self.data_args.cutoff_len:
                break

            source_len, target_len = infer_seqlen(
                len(source_ids), len(target_ids), self.data_args.cutoff_len - total_length
            )
            source_ids = source_ids[:source_len]
            target_ids = target_ids[:target_len]
            total_length += source_len + target_len

            if self.data_args.train_on_prompt:
                source_label = source_ids
            elif self.template.efficient_eos:
                source_label = [self.tokenizer.eos_token_id] + [IGNORE_INDEX] * (source_len - 1)
            else:
                source_label = [IGNORE_INDEX] * source_len

            if self.data_args.mask_history and turn_idx != 0:  # train on the last turn only
                target_label = [IGNORE_INDEX] * target_len
            else:
                target_label = target_ids

            if self.data_args.mask_history:  # reversed sequences
                input_ids = source_ids + target_ids + input_ids
                labels = source_label + target_label + labels
            else:
                input_ids += source_ids + target_ids
                labels += source_label + target_label

        if self.template.efficient_eos:
            input_ids += [self.tokenizer.eos_token_id]
            labels += [self.tokenizer.eos_token_id]

        # 应用 MFT (Masked Fine-Tuning)
        # 假设 self.data_args 中增加了以下属性:
        # self.data_args.use_mft: bool (例如 True)
        # self.data_args.mft_mask_probability: float (例如 0.15)
        # self.data_args.mft_mask_token_strategy: str (例如 "pad_token" 或 "mask_token")
        # if hasattr(self.data_args, "use_mft") and self.data_args.use_mft:
        #     mask_idx: int
        #     mft_strategy = getattr(self.data_args, "mft_mask_token_strategy", "pad_token")
        #     mft_prob = getattr(self.data_args, "mft_mask_probability", 0.20)

        #     if mft_strategy == "mask_token":
        #         if self.tokenizer.mask_token_id is not None:
        #             mask_idx = self.tokenizer.mask_token_id
        #         else:
        #             logger.warning_once(
        #                 "MFT: mft_mask_token_strategy is 'mask_token' but tokenizer has no mask_token_id. "
        #                 "Falling back to pad_token_id."
        #             )
        #             if self.tokenizer.pad_token_id is None:
        #                 raise ValueError(
        #                     "MFT: Tokenizer must have a pad_token_id for MFT fallback when mask_token is unavailable."
        #                 )
        #             mask_idx = self.tokenizer.pad_token_id
        #     elif mft_strategy == "pad_token":
        #         if self.tokenizer.pad_token_id is None:
        #             raise ValueError("MFT: Tokenizer must have a pad_token_id for MFT when strategy is 'pad_token'.")
        #         mask_idx = self.tokenizer.pad_token_id
        #     else:
        #         raise ValueError(f"MFT: Unknown mft_mask_token_strategy: {mft_strategy}")

        #     if mft_prob > 0: # 仅当掩码概率大于0时执行
        #         input_ids = _apply_mft_masking(
        #             input_ids_single_example=input_ids,
        #             labels_single_example=labels,
        #             mask_probability=mft_prob,
        #             mask_index=mask_idx,
        #             ignore_index=IGNORE_INDEX
        #         )

        return input_ids, labels

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
        # for multiturn examples, we only mask the prompt part in each prompt-response pair.
        model_inputs = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue

            input_ids, labels = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
                current_index=i, # 传递当前索引 i
            )
            length = len(input_ids)
            if self.data_args.drop_exceed_length_data and length >= self.data_args.cutoff_len:
                logger.warning_rank0(f"Dropped lengthy example with length {length} > {self.data_args.cutoff_len}.")
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["position_ids"].append(list(range(len(input_ids))))#新增position_ids处理
            model_inputs["labels"].append(labels)
            model_inputs["images"].append(examples["_images"][i])
            model_inputs["videos"].append(examples["_videos"][i])
            model_inputs["audios"].append(examples["_audios"][i])
            logger.info(f"Successfully tokenized and processed original_index: {i} for SupervisedDatasetProcessor.") # 新增日志

        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["labels"]))
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("label_ids:\n{}".format(example["labels"]))
        print(f"labels:\n{self.tokenizer.decode(valid_labels, skip_special_tokens=False)}")


@dataclass
class PackedSupervisedDatasetProcessor(SupervisedDatasetProcessor):
    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # TODO: use `position_ids` to achieve packing
        # build inputs with format `<bos> X1 Y1 <eos> <bos> X2 Y2 <eos>`
        # and labels with format `<ignore> ... <ignore> Y1 <eos> <ignore> ... <ignore> Y2 <eos>`
        valid_num = 0
        batch_input_ids, batch_labels, batch_images, batch_videos, batch_audios = [], [], [], [], []
        lengths = []
        length2indexes = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue

            input_ids, labels = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
                current_index=i, # 传递当前索引 i
            )
            length = len(input_ids)
            if length > self.data_args.cutoff_len:
                logger.warning_rank0(f"Dropped lengthy example with length {length} > {self.data_args.cutoff_len}.")
            else:
                lengths.append(length)
                length2indexes[length].append(valid_num)
                batch_input_ids.append(input_ids)
                batch_labels.append(labels)
                batch_images.append(examples["_images"][i] or [])
                batch_videos.append(examples["_videos"][i] or [])
                batch_audios.append(examples["_audios"][i] or [])
                valid_num += 1

        model_inputs = defaultdict(list)
        #新增package策略选择
        if self.data_args.packing_method == "greedy":
            knapsacks = greedy_knapsack(lengths, self.data_args.cutoff_len - 1)  # reserved for the padding token
        elif self.data_args.packing_method == "random":
            knapsacks = random_knapsack(lengths, self.data_args.cutoff_len - 1)  # reserved for the padding token
        else:
            raise ValueError(f"The pack method {self.data_args.pack_method} not supported!!")
        for knapsack_idx, knapsack in enumerate(knapsacks):
            packed_input_ids, packed_attention_masks, packed_position_ids, packed_labels = [], [], [], []
            packed_images, packed_videos, packed_audios = [], [], []
            for i, length in enumerate(knapsack):
                index = length2indexes[length].pop()
                packed_input_ids += batch_input_ids[index]
                packed_position_ids += list(range(len(batch_input_ids[index])))  # NOTE: pad_to_multiple_of ignore this
                packed_labels += batch_labels[index]
                packed_images += batch_images[index]
                packed_videos += batch_videos[index]
                packed_audios += batch_audios[index]
                if self.data_args.neat_packing:
                    packed_attention_masks += [i + 1] * len(batch_input_ids[index])  # start from 1
                else:
                    packed_attention_masks += [1] * len(batch_input_ids[index])

            logger.info(f"Knapsack {knapsack_idx}: About to pack original_indices") # 修改日志内容，表明即将打包
            if len(packed_input_ids) < self.data_args.cutoff_len + 1:  # avoid flash_attn drops attn mask
                pad_length = self.data_args.cutoff_len - len(packed_input_ids) + 1
                packed_input_ids += [self.tokenizer.pad_token_id] * pad_length
                packed_position_ids += [0] * pad_length
                packed_labels += [IGNORE_INDEX] * pad_length
                if self.data_args.neat_packing:
                    packed_attention_masks += [0] * pad_length
                else:
                    packed_attention_masks += [1] * pad_length  # more efficient flash_attn

            if len(packed_input_ids) != self.data_args.cutoff_len + 1:
                raise ValueError("The length of packed example should be identical to the cutoff length.")

            model_inputs["input_ids"].append(packed_input_ids)
            model_inputs["attention_mask"].append(packed_attention_masks)
            model_inputs["position_ids"].append(packed_position_ids)
            model_inputs["labels"].append(packed_labels)
            model_inputs["images"].append(packed_images or None)
            model_inputs["videos"].append(packed_videos or None)
            model_inputs["audios"].append(packed_audios or None)
            logger.info(f"Successfully tokenized and packed knapsack {knapsack_idx} containing original_indices.") # 新增日志

        return model_inputs
