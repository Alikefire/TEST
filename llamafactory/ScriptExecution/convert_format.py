import json
import os

def load_filter_instructions(filter_json_path):
    """Loads instructions from the filter JSON file."""
    filter_instructions = set()
    try:
        with open(filter_json_path, 'r', encoding='utf-8') as f_filter:
            filter_data = json.load(f_filter)
            if isinstance(filter_data, list):
                for item in filter_data:
                    if isinstance(item, dict) and "instruction" in item:
                        filter_instructions.add(item["instruction"])
            elif isinstance(filter_data, dict) and "instruction" in filter_data:
                 filter_instructions.add(filter_data["instruction"])
    except FileNotFoundError:
        print(f"Error: Filter file not found at {filter_json_path}")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from filter file {filter_json_path}. Ensure it's valid JSON.")
    return filter_instructions

def convert_and_filter_jsonl(input_jsonl_path, filter_instructions_set, output_jsonl_path):
    """
    Converts records from input_jsonl_path to a new format and filters them
    based on a provided set of filter instructions.
    Returns a set of instructions that were actually written to the output.
    """
    written_instructions = set()
    converted_count = 0
    try:
        with open(input_jsonl_path, 'r', encoding='utf-8') as f_in, \
             open(output_jsonl_path, 'w', encoding='utf-8') as f_out:
            for line_number, line in enumerate(f_in, 1):
                try:
                    record = json.loads(line)
                    user_content = None
                    assistant_content = None

                    if "messages" in record and isinstance(record["messages"], list):
                        for message in record["messages"]:
                            if message.get("role") == "user" and "content" in message:
                                user_content = message["content"]
                            elif message.get("role") == "assistant" and "content" in message:
                                assistant_content = message["content"]
                        
                        if user_content and assistant_content:
                            if user_content in filter_instructions_set:
                                new_record = {
                                    "instruction": user_content,
                                    "input": "",
                                    "output": assistant_content
                                }
                                f_out.write(json.dumps(new_record, ensure_ascii=False) + '\n')
                                written_instructions.add(user_content)
                                converted_count += 1
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line {line_number} in input file: {line.strip()}")
                except Exception as e:
                    print(f"An error occurred processing line {line_number} from {input_jsonl_path}: {e}")
        
        print(f"Initial conversion complete. {converted_count} records written to {output_jsonl_path} based on filter set.")
        return written_instructions, converted_count

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_jsonl_path}")
        return set(), 0
    except Exception as e:
        print(f"An unexpected error occurred during initial conversion: {e}")
        return set(), 0

def supplement_output_from_input(input_jsonl_path, existing_instructions_set, target_total_instructions, output_jsonl_path):
    """
    Supplements the output file with additional unique records from the input file.

    Args:
        input_jsonl_path (str): Path to the input JSONL file (format 1).
        existing_instructions_set (set): A set of instructions already processed/written or from the filter file.
        target_total_instructions (int): The desired total number of unique instructions.
        output_jsonl_path (str): Path to the output JSONL file to append to.
    """
    supplemented_count = 0
    # Make a copy to track newly added instructions in this function call, to avoid re-adding within the same call
    current_instructions_in_output = set(existing_instructions_set) 

    if len(current_instructions_in_output) >= target_total_instructions:
        print("No supplementation needed. Current instruction count meets or exceeds target.")
        return 0

    try:
        with open(input_jsonl_path, 'r', encoding='utf-8') as f_in, \
             open(output_jsonl_path, 'a', encoding='utf-8') as f_out: # Append mode
            for line_number, line in enumerate(f_in, 1):
                if len(current_instructions_in_output) >= target_total_instructions:
                    break # Stop if we've reached the target
                try:
                    record = json.loads(line)
                    user_content = None
                    assistant_content = None

                    if "messages" in record and isinstance(record["messages"], list):
                        for message in record["messages"]:
                            if message.get("role") == "user" and "content" in message:
                                user_content = message["content"]
                            elif message.get("role") == "assistant" and "content" in message:
                                assistant_content = message["content"]
                        
                        if user_content and assistant_content:
                            # Add if the user_content is not in the existing set of instructions
                            if user_content not in current_instructions_in_output:
                                new_record = {
                                    "instruction": user_content,
                                    "input": "",
                                    "output": assistant_content
                                }
                                f_out.write(json.dumps(new_record, ensure_ascii=False) + '\n')
                                current_instructions_in_output.add(user_content) # Add to our tracking set
                                supplemented_count += 1
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line {line_number} during supplementation: {line.strip()}")
                except Exception as e:
                    print(f"An error occurred processing line {line_number} from {input_jsonl_path} during supplementation: {e}")
        
        print(f"Supplementation complete. {supplemented_count} additional records written to {output_jsonl_path}.")
        return supplemented_count

    except FileNotFoundError:
        print(f"Error: Input file for supplementation not found at {input_jsonl_path}")
        return 0
    except Exception as e:
        print(f"An unexpected error occurred during supplementation: {e}")
        return 0

if __name__ == '__main__':
    input_file = "/home/xiexin/xx_help/LLaMA-Factory/data/N8Programs/gsm8k-r1/traces.jsonl"
    filter_file = "/home/xiexin/xx_help/LLaMA-Factory/data/N8Programs/gsm8k-gpt4o/train.json"
    output_file = "/home/xiexin/xx_help/LLaMA-Factory/data/N8Programs/gsm8k-r1/train_supplemented.jsonl"

    # Create dummy input files for testing if they don't exist
    if not os.path.exists(input_file):
        print(f"Creating dummy input file: {input_file}")
        with open(input_file, 'w', encoding='utf-8') as f:
            f.write('{"messages":[{"role":"user","content":"Instruction A from input (in filter)"},{"role":"assistant","content":"Assistant for A"}]}\n')
            f.write('{"messages":[{"role":"user","content":"Instruction B from input (in filter)"},{"role":"assistant","content":"Assistant for B"}]}\n')
            f.write('{"messages":[{"role":"user","content":"Instruction C from input (unique)"},{"role":"assistant","content":"Assistant for C"}]}\n')
            f.write('{"messages":[{"role":"user","content":"Instruction D from input (unique)"},{"role":"assistant","content":"Assistant for D"}]}\n')
            f.write('{"messages":[{"role":"user","content":"Instruction E from input (unique)"},{"role":"assistant","content":"Assistant for E"}]}\n')

    if not os.path.exists(filter_file):
        print(f"Creating dummy filter file: {filter_file}")
        with open(filter_file, 'w', encoding='utf-8') as f:
            json.dump([
                {"instruction": "Instruction A from input (in filter)", "input": "", "output": "Output for A from filter"},
                {"instruction": "Instruction B from input (in filter)", "input": "", "output": "Output for B from filter"},
                {"instruction": "Instruction X from filter (not in input)", "input": "", "output": "Output for X"},
                {"instruction": "Instruction Y from filter (not in input)", "input": "", "output": "Output for Y"}
            ], f, ensure_ascii=False, indent=4)

    print(f"Starting processing...\nInput: {input_file}\nFilter: {filter_file}\nOutput: {output_file}")

    # 1. Load instructions from the filter file
    original_filter_instructions = load_filter_instructions(filter_file)
    target_instruction_count = len(original_filter_instructions)
    print(f"Target number of unique instructions from filter file: {target_instruction_count}")

    if target_instruction_count == 0:
        print("No instructions in filter file. Exiting.")
    else:
        # 2. Perform initial conversion based on filter_file instructions
        # This will create/overwrite the output_file
        written_instructions_after_conversion, initial_converted_count = convert_and_filter_jsonl(
            input_file, 
            original_filter_instructions, 
            output_file
        )
        print(f"Number of unique instructions after initial conversion: {len(written_instructions_after_conversion)}")

        # 3. Supplement if needed
        # The existing_instructions_set for supplementation should include both 
        # what was in the original filter AND what was just written (if any from input matched the filter).
        # However, to ensure we only add *new* items from input_file not in original_filter_instructions,
        # we pass original_filter_instructions to the supplement function's existing_instructions_set argument.
        # The supplement function will then only pick items from input_file whose instructions are not in original_filter_instructions.
        
        # We need to know all instructions that are *currently* in the output file or *should be* (from filter)
        # to avoid adding duplicates during supplementation.
        # The `written_instructions_after_conversion` already contains instructions from `input_file` that matched `original_filter_instructions`.
        # We also need to consider instructions from `original_filter_instructions` that might *not* have been in `input_file`.
        # The most robust set of "already accounted for" instructions is `original_filter_instructions` plus anything *newly* added from `input_file`.
        # For supplementation, we want to pick from input_file items whose instructions are NOT in original_filter_instructions.
        # The `supplement_output_from_input` function's `existing_instructions_set` should be the set of instructions we don't want to pick again.
        # So, it should be `original_filter_instructions`.
        
        # Let's refine: the `existing_instructions_set` for supplementation should be the set of all instructions
        # that we consider "covered", which are those in `original_filter_instructions`.
        # The `supplement_output_from_input` will then pick from `input_file` only those instructions
        # that are NOT in `original_filter_instructions` until the `target_instruction_count` is met.

        # The set of instructions already in the output file (or intended to be from filter)
        all_known_instructions = written_instructions_after_conversion 
        # # Add instructions that were actually written from input_file matching the filter
        # all_known_instructions.update(written_instructions_after_conversion) 

        if len(all_known_instructions) < target_instruction_count:
            print(f"Need to supplement. Current unique instructions: {len(all_known_instructions)}, Target: {target_instruction_count}")
            supplement_needed_count = target_instruction_count - len(all_known_instructions)
            
            # For supplementation, we want to pick items from input_file whose instructions are NOT in `all_known_instructions`.
            supplement_output_from_input(
                input_file, 
                all_known_instructions, # Pass all instructions we already know about (from filter or initial conversion)
                target_instruction_count, 
                output_file
            )
        else:
            print("Sufficient instructions after initial conversion. No supplementation needed.")

        # Final check of the output file
        final_instructions_in_output = set()
        final_record_count = 0
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f_final:
                for line in f_final:
                    try:
                        record = json.loads(line)
                        final_instructions_in_output.add(record['instruction'])
                        final_record_count +=1
                    except json.JSONDecodeError:
                        pass # Ignore errors in final check, already handled
        print(f"Processing finished. Total records in output file '{output_file}': {final_record_count}")
        print(f"Total unique instructions in output file: {len(final_instructions_in_output)}")