# WoTE 昇腾训练说明

训练入口通过 `torch_npu` 和 PyTorch Lightning 接入昇腾 NPU。已测试的环境为
CANN 8.5.2、PyTorch 2.7.1、torch_npu 2.7.1.post2 和 Lightning 2.2.1；本机
`npu-smi` 将设备识别为 Ascend910。已验证单卡和单机双卡 FP32 训练，以及单卡和双卡推理；
更多卡数和混合精度尚未验证。

在本机选择物理设备并激活现有环境：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
source /home/gaoxuan_gac/model_setup/wote_setup/activate_wote_npu.sh
cd /home/gaoxuan_gac/code/WoTE
```

运行 `navsim/planning/script/run_training.py` 时使用 `trainer.params.accelerator=npu`、
`trainer.params.strategy=auto`、`trainer.params.precision=32-true` 和
`+trainer.params.devices=1`。`ASCEND_RT_VISIBLE_DEVICES` 选择的物理设备在进程内映射为
`npu:0`。训练还需要 NAVTRAIN 传感器文件、`WoTEConfig` 中配置的 256 条轨迹锚点，以及覆盖训练和验证
场景的 metric cache。每次前向在 CPU 上对当前 refine 轨迹按 4 秒、40 步在线计算 PDM 五项标签，
不再读取预计算锚点分数；在线仿真会增加训练耗时。训练前将环境变量
`WOTE_METRIC_CACHE_PATH` 设置为包含 `metadata` 的 metric cache 根目录，或配置
`agent.config.metric_cache_path`。缺少场景缓存时训练会报错。损失函数中的 SciPy 匈牙利匹配也在
CPU 上运行。

在线评分默认每张 NPU 使用 16 个独立 CPU 评分进程，启用批量轨迹转换，并在每个进程内
最多缓存 128 个场景。该配置按本机 mini 数据的八卡、每卡 batch 16 测速选出；其他机器或
批大小应先核对 CPU 与内存资源。可通过 `agent.config.score_workers`、
`agent.config.score_fast_trajectory`、`agent.config.score_cache_size` 调整，环境变量
`WOTE_SCORE_WORKERS`、`WOTE_SCORE_FAST`、`WOTE_SCORE_CACHE_SIZE` 优先；将进程数设为 0
可回退原串行评分。场景缓存只保留不随模型变化的信息，每次仍对当前 refine 候选在线评分。

未来 BEV 标签使用 `WoTE_target_refined_v1` 缓存中的无自车框背景，在训练时按采样的 refine
轨迹动态绘制自车框。旧 `WoTE_target.gz` 不能用作新标签；使用缓存时需重新生成目标缓存。
非强制重建模式下，已有的传感器特征缓存会复用。缓存模式发现旧目标时会提示重建。
下方测试命令同样要求指定的训练和验证场景均有对应 metric cache。

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

## 单机多卡训练

先用 `npu-smi info` 确认设备空闲。在启动进程前设置可见物理设备，例如
`export ASCEND_RT_VISIBLE_DEVICES=2,3`；进程内映射为 `npu:0` 和 `npu:1`。
在训练命令中设置：

```bash
trainer.params.accelerator=npu trainer.params.strategy=ddp \
trainer.params.precision=32-true +trainer.params.devices=2
```

训练采用每卡一个进程的 HCCL DDP；`dataloader.params.batch_size` 是**每卡**批大小，
全局批大小为每卡批大小乘设备数，再乘梯度累积步数。改变全局批大小可能改变训练效果，
不应将速度提升直接视作同等训练配置。多卡验证使用 Lightning 的分布式采样，验证集大小不能
被卡数整除时可能重复少量样本；最终场景评分应使用下方推理流程。

要恢复包含优化器、学习率调度器和轮次的 Lightning 检查点，可在训练命令中追加
`resume_from_checkpoint='"/绝对路径/epoch=0-step=1274.ckpt"'`。`trainer.params.max_epochs`
是训练总轮数，不是恢复后新增的轮数。恢复时应保持模型和优化器配置兼容，并使用新的输出目录
保存新运行；检查点只保存到最近一次完成的保存点。

## 单机多卡推理与 PDMS

同样先设置 `ASCEND_RT_VISIBLE_DEVICES`，然后在 `run_pdm_score.py` 的常规参数中追加：

```bash
'inference_devices=["npu:0","npu:1"]'
```

每张卡独立处理不重叠的场景，主进程核验场景完整性后合并 CSV；各场景的模型输出仍保存到
`output_dir/model_results`。此参数不能与单卡参数 `+single_eval=true` 或
`+inference_device=npu:0` 同时使用。子进程各自加载模型和场景，因此实际加速取决于场景数量、
数据读取和 CPU 评分耗时。
