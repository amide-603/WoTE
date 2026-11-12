"""为 Lightning 训练入口提供昇腾 NPU 支持。"""

import torch
import torch_npu  # noqa: F401  # 导入时向 PyTorch 注册 NPU 设备类型。
from pytorch_lightning.accelerators import Accelerator
from pytorch_lightning.strategies import DDPStrategy
from torch.nn.parallel import DistributedDataParallel


class NPUAccelerator(Accelerator):
    """使用 ASCEND_RT_VISIBLE_DEVICES 所选择的逻辑 NPU。"""

    def setup_device(self, device):
        # 检查并设置当前 NPU，避免训练被错误地放到 CPU 或其他设备上。
        if device.type != "npu":
            raise ValueError(f"Expected an NPU device, got {device}")
        torch.npu.set_device(device)

    def get_device_stats(self, device):
        # 当前训练未接入设备统计采集，返回空字典以满足 Lightning 接口。
        return {}

    def teardown(self):
        pass

    @staticmethod
    def parse_devices(devices):
        if devices == "auto":
            return 1
        if isinstance(devices, str) and devices.isdigit():
            devices = int(devices)
        if isinstance(devices, bool) or not isinstance(devices, int) or devices < 1:
            raise ValueError("NPU devices must be a positive device count or 'auto'")
        available = torch.npu.device_count()
        if devices > available:
            raise ValueError(f"Requested {devices} NPUs, but only {available} are visible")
        return devices

    @staticmethod
    def get_parallel_devices(devices):
        count = NPUAccelerator.parse_devices(devices)
        return [torch.device(f"npu:{index}") for index in range(count)]

    @staticmethod
    def auto_device_count():
        return 1

    @staticmethod
    def is_available():
        # 让 Lightning 在启动训练前检查 torch_npu 是否能访问设备。
        return torch.npu.is_available()


class NPUDDPStrategy(DDPStrategy):
    """Avoid Lightning's CUDA stream when wrapping a single-device NPU model."""

    def _setup_model(self, model):
        return DistributedDataParallel(
            module=model, device_ids=[self.root_device.index], **self._ddp_kwargs
        )
