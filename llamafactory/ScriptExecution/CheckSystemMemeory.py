import torch
import gc
import psutil

def check_system_memory():
    vm = psutil.virtual_memory()
    print(f"\n系统内存使用情况:")
    print(f"总内存: {vm.total/1024**3:.2f}GB")
    print(f"可用内存: {vm.available/1024**3:.2f}GB")
    print(f"内存使用率: {vm.percent}%")
    print(f"已用内存: {vm.used/1024**3:.2f}GB")
    print(f"空闲内存: {vm.free/1024**3:.2f}GB")

check_system_memory()