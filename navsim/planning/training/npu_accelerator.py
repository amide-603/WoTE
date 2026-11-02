"""为 Lightning 训练入口提供单卡昇腾 NPU 支持。"""

import torch
import torch_npu  # noqa: F401  # 导入时向 PyTorch 注册 NPU 设备类型。
from pytorch_lightning.accelerators import Accelerator


class NPUAccelerator(Accelerator):
    """通过 Lightning 自定义加速器使用第一个可见 NPU，不启用多卡。"""

    def setup_device(self, device):
        # 检查并设置当前 NPU，避免训练被错误地放到 CPU 或其他设备上。
        if device.type != "npu":
            raise ValueError(f"Expected an NPU device, got {device}")
        torch.npu.set_device(device)

    def get_device_stats(self, device):
        # 当前训练未接入设备统计采集，返回空字典以满足 Lightning 接口。
        return {}

    def teardown(self):
        # 单设备训练无需额外的加速器清理步骤。
        pass

    @staticmethod
    def parse_devices(devices):
        # 多卡通信尚未适配和验证，只接受一个可见设备。
        if devices not in (1, "1", "auto"):
            raise ValueError("NPU training supports exactly one visible device")
        return 1

    @staticmethod
    def get_parallel_devices(devices):
        NPUAccelerator.parse_devices(devices)
        # 物理卡由 ASCEND_RT_VISIBLE_DEVICES 选择，进程内使用逻辑编号 0。
        return [torch.device("npu:0")]

    @staticmethod
    def auto_device_count():
        # 自动选择时也保持单卡，避免意外使用所有可见 NPU。
        return 1

    @staticmethod
    def is_available():
        # 让 Lightning 在启动训练前检查 torch_npu 是否能访问设备。
        return torch.npu.is_available()
