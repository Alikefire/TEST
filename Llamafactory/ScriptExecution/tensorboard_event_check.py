from tensorboard.backend.event_processing import event_accumulator
import json # 用于美化打印字典

def print_event_info(event_file):
    ea = event_accumulator.EventAccumulator(
        event_file,
        size_guidance={  # 加载所有数据
            event_accumulator.COMPRESSED_HISTOGRAMS: 0,
            event_accumulator.IMAGES: 0,
            event_accumulator.AUDIO: 0,
            event_accumulator.SCALARS: 0,
            event_accumulator.HISTOGRAMS: 0,
            event_accumulator.TENSORS: 0, # 尝试加载张量，HParams和文本摘要可能在这里
            # event_accumulator.TEXT: 0,    # 移除此行，因为 TEXT 不是有效的 size_guidance键
        }
    )
    ea.Reload() # 加载事件

    print("="*80)
    print(f"事件文件: {event_file}")
    
    tags = ea.Tags()
    print("\n所有可用 Tags 类别：")
    for k, v_list in tags.items():
        if isinstance(v_list, (list, tuple)):
            print(f"  类别 '{k}': 包含 {len(v_list)} 个 tags")
        else: # 例如 graph, meta_graph 可能是布尔值
             print(f"  类别 '{k}': {v_list}")

    # 检查 Text Summaries (HuggingFace Trainer 通常在这里记录参数)
    print("\nText Summaries (检查 'args' 或类似参数记录):")
    text_tags = tags.get('text', [])
    if not text_tags:
        print("  未找到任何 Text Summaries。")
    for tag in text_tags:
        text_events = ea.Text(tag)
        print(f"  Tag '{tag}': 共 {len(text_events)} 条记录")
        for i, event in enumerate(text_events[:1]): # 只打印第一条记录的详细内容以避免过长输出
            print(f"    记录 {i+1}:")
            print(f"      Wall time: {event.wall_time}")
            print(f"      Step: {event.step}")
            # TextEvent 的 value 是 bytes, 需要 decode
            try:
                # 尝试将文本内容解析为JSON，因为HuggingFace参数可能是JSON字符串
                value_str = event.tensor_proto.string_val[0].decode('utf-8')
                print(f"      Value (原始): {value_str[:200]}...") # 打印部分原始值
                if tag == 'args' or 'hparams' in tag or 'params' in tag : # 如果是参数相关的tag
                    try:
                        params_dict = json.loads(value_str)
                        print(f"      Value (解析为JSON后美化打印): \n{json.dumps(params_dict, indent=2, ensure_ascii=False)}")
                    except json.JSONDecodeError:
                        print(f"      Value (无法解析为JSON): {value_str}")
            except Exception as e:
                print(f"      Value (解码或处理时出错): {e}")
                print(f"      Raw tensor_proto: {event.tensor_proto}")


    # 检查 Scalars (通常用于 loss, learning_rate 等)
    print("\nScalars:")
    scalar_tags = tags.get('scalars', [])
    if not scalar_tags:
        print("  未找到任何 Scalars。")
    for tag in scalar_tags:
        scalar_events = ea.Scalars(tag)
        print(f"  Tag '{tag}': 共 {len(scalar_events)} 条记录, 前3条: {scalar_events[:3]}")
    
    # 提示HParams插件
    print("\n关于 HParams 插件:")
    print("  Hugging Face Trainer 使用 add_hparams 记录参数，这些参数主要通过 TensorBoard UI 的 'HPARAMS' 标签页查看。")
    print("  如果 'HPARAMS' 标签页为空，请确认 Trainer 的 report_to 设置，并检查 'TEXT' 标签页中是否有 'args' 相关的记录。")

    print("="*80 + "\n")

if __name__ == '__main__':
    # 请替换为你的实际事件文件路径
    event_file_1 = "/home/zdd/xx_help/MaskedThought/log_output/tensorboard_log_20250509_1955/events.out.tfevents.1746789140.dell-PowerEdge-T640.3210210.0"
    event_file_2 = "/home/zdd/xx_help/MaskedThought/log_output/tensorboard_log/events.out.tfevents.1746791855.dell-PowerEdge-T640.3271818.0" # 示例的第二个文件

    print(f"正在检查文件: {event_file_1}")
    print_event_info(event_file_1)
    
    # 如果你有第二个文件需要检查，取消下面的注释
    print(f"正在检查文件: {event_file_2}")
    print_event_info(event_file_2)