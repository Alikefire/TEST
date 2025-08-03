from torch.utils.tensorboard import SummaryWriter
import time
import os

# 1. 创建一个 SummaryWriter 实例，指定日志目录
# 日志目录将是 ./log_output/runs/tensorboard_test_from_terminal_ 当前时间戳
# 确保 ./log_output/runs 目录存在
log_base_dir = "./log_output/runs"
os.makedirs(log_base_dir, exist_ok=True)
log_dir = os.path.join(log_base_dir, f"tensorboard_test_from_terminal_{int(time.time())}")

writer = SummaryWriter(log_dir=log_dir)
print(f"TensorBoard 日志已写入到: {log_dir}")

# 2. 添加一些通用的超参数文本 (可选，但有助于测试)
hparams_general = {"script_purpose": "Test logging terminal parameters", "test_type": "manual_log"}
hparams_text_general = "\n".join([f"{key}: {value}" for key, value in hparams_general.items()])
writer.add_text('general_info', hparams_text_general, 0)

# 3. 记录从终端日志中提取的参数 (以 step 1399 为例)
step = 1399

# 来自 {'loss': 0.8821, 'learning_rate': 9.399081388794688e-06, 'epoch': 1.71}
# 注意：这个 'loss' 可能与 trainer_436.py 中的 'his_loss' 不同，这里我们都记录下来以作区分
writer.add_scalar('train/loss_overall', 0.8821, step)
writer.add_scalar('train/learning_rate_overall', 9.399081388794688e-06, step)
writer.add_scalar('train/epoch', 1.71, step)

# 来自 [INFO|trainer_436.py:...] @ step 1399
writer.add_scalar('custom_metrics/his_ce_loss', 0.7841649653017521, step)
writer.add_scalar('custom_metrics/his_loss', 0.7841649653017521, step) # 这与上面的 'train/loss' 可能不同

# gram_acc: 2:0.6935386657714844, 3:0.6225000023841858, 4:0.599056601524353
writer.add_scalar('custom_metrics/gram_acc/2', 0.6935386657714844, step)
writer.add_scalar('custom_metrics/gram_acc/3', 0.6225000023841858, step)
writer.add_scalar('custom_metrics/gram_acc/4', 0.599056601524353, step)

# gram_loss: 2:0.4374343752861023, 3:0.5328543782234192, 4:0.5454993844032288
writer.add_scalar('custom_metrics/gram_loss/2', 0.4374343752861023, step)
writer.add_scalar('custom_metrics/gram_loss/3', 0.5328543782234192, step)
writer.add_scalar('custom_metrics/gram_loss/4', 0.5454993844032288, step)

# gram_ent: 1: 1.366059422492981, 2:1.094343900680542, 3:1.3156489133834839, 4:1.4294618368148804
writer.add_scalar('custom_metrics/gram_ent/1', 1.366059422492981, step)
writer.add_scalar('custom_metrics/gram_ent/2', 1.094343900680542, step)
writer.add_scalar('custom_metrics/gram_ent/3', 1.3156489133834839, step)
writer.add_scalar('custom_metrics/gram_ent/4', 1.4294618368148804, step)

writer.add_scalar('custom_metrics/mask_rate', 0.10644403861205323, step)
writer.add_scalar('custom_metrics/lr_trainer436', 9.372803676728138e-06, step) # 与 overall lr 可能略有不同或来源不同

# 你可以仿照上面的方式，添加更多步骤和其他参数
# 例如，模拟 step 1599
step_1599 = 1599
writer.add_scalar('custom_metrics/his_ce_loss', 0.7793896825611591, step_1599)
writer.add_scalar('custom_metrics/his_loss', 0.7793896825611591, step_1599)
writer.add_scalar('custom_metrics/mask_rate', 0.11270545264805636, step_1599)
writer.add_scalar('custom_metrics/lr_trainer436', 9.290839735824254e-06, step_1599)
# ... 添加 gram_acc, gram_loss, gram_ent for step 1599 ...

# 4. 关闭 writer
writer.close()

print("测试数据已写入。现在你可以使用 tensorboard 命令来查看。")
print(f"请在终端运行: tensorboard --logdir {log_dir}")
print(f"或者，如果你想查看所有 {log_base_dir} 下的日志: tensorboard --logdir {log_base_dir}")