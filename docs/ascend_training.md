# WoTE 昇腾训练说明

训练入口通过 `torch_npu` 和 PyTorch Lightning 接入单卡昇腾 NPU。已测试的环境为
CANN 8.5.2、PyTorch 2.7.1、torch_npu 2.7.1.post2 和 Lightning 2.2.1；本机
`npu-smi` 将设备识别为 Ascend910。目前仅支持已验证的单卡 FP32 训练，多卡和混合精度尚未验证。

在本机选择物理设备并激活现有环境：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
source /home/gaoxuan_gac/model_setup/wote_setup/activate_wote_npu.sh
cd /home/gaoxuan_gac/code/WoTE
```

运行 `navsim/planning/script/run_training.py` 时使用 `trainer.params.accelerator=npu`、
`trainer.params.strategy=auto`、`trainer.params.precision=32-true` 和
`+trainer.params.devices=1`。`ASCEND_RT_VISIBLE_DEVICES` 选择的物理设备在进程内映射为
`npu:0`。训练还需要 NAVTRAIN 传感器文件，以及 `WoTEConfig` 中配置的轨迹锚点和预计算仿真奖励。
损失函数中的 SciPy 匈牙利匹配在 CPU 上运行。

使用真实 NAVTRAIN 样本进行有范围限制的小规模测试：

```bash
python navsim/planning/script/run_training.py \
  agent=WoTE_agent experiment_name=npu_train_smoke split=trainval \
  scene_filter=all_scenes scene_filter.frame_interval=1 scene_filter.max_scenes=2 \
  'scene_filter.tokens=[1aa44d46e4ab5bc7,2570fbfdf1835706,1bf6fef253f45586,b3c794a291025583]' \
  'train_logs=[2021.05.12.19.36.12_veh-35_00005_00204]' \
  'val_logs=[2021.06.07.11.59.52_veh-35_02283_02464]' \
  dataloader.params.batch_size=2 dataloader.params.num_workers=0 \
  dataloader.params.prefetch_factor=null dataloader.params.pin_memory=false \
  trainer.params.accelerator=npu trainer.params.strategy=auto \
  trainer.params.precision=32-true +trainer.params.devices=1 \
  trainer.params.max_epochs=1 trainer.params.check_val_every_n_epoch=1 \
  trainer.params.limit_train_batches=1 trainer.params.limit_val_batches=1 \
  +trainer.params.log_every_n_steps=1 \
  output_dir=/home/gaoxuan_gac/model_setup/wote_setup/npu_train_smoke
```

此命令分别运行一个训练批次和一个验证批次，每批包含两个场景；生成的检查点仅用于验证训练链路，
并非训练完成的模型。Hydra 和 Lightning 将测试结果保存到指定的 `output_dir`。本机普通
`trainval` 划分中有些日志缺少传感器文件，选择其他样本时应使用 NAVTRAIN 场景 token，
或先核对传感器文件是否齐全。
