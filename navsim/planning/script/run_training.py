from typing import Tuple
import hydra
from hydra.utils import instantiate
import logging, torch
from omegaconf import DictConfig
from pathlib import Path
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from navsim.planning.training.dataset import CacheOnlyDataset, Dataset
from navsim.planning.training.agent_lightning_module import AgentLightningModule
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter
from navsim.agents.abstract_agent import AbstractAgent

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"

def build_datasets(cfg: DictConfig, agent: AbstractAgent) -> Tuple[Dataset, Dataset]:
    train_scene_filter: SceneFilter = instantiate(cfg.scene_filter)
    if train_scene_filter.log_names is not None:
        # train_scene_filter.log_names = [l for l in train_scene_filter.log_names if l in cfg.train_logs]
        train_scene_filter.log_names = list(set(train_scene_filter.log_names) & set(cfg.train_logs))
    else:
        train_scene_filter.log_names = cfg.train_logs

    val_scene_filter: SceneFilter = instantiate(cfg.scene_filter)
    if val_scene_filter.log_names is not None:
        # val_scene_filter.log_names = [l for l in val_scene_filter.log_names if l in cfg.val_logs]
        val_scene_filter.log_names = list(set(val_scene_filter.log_names) & set(cfg.val_logs))
    else:
        val_scene_filter.log_names = cfg.val_logs

    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)
    train_debug = cfg.train_debug if hasattr(cfg, "train_debug") else False

    train_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=train_scene_filter,
        sensor_config=agent.get_sensor_config(),
        train_debug=train_debug,
    )

    val_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=val_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    use_fut_frames = agent.config.use_fut_frames if hasattr(agent.config, "use_fut_frames") else False
    train_data = Dataset(
        scene_loader=train_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
        use_fut_frames=use_fut_frames,
    )

    val_data = Dataset(
        scene_loader=val_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
        use_fut_frames=use_fut_frames,
    )

    return train_data, val_data


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    logger.info("Global Seed set to 0")
    pl.seed_everything(0, workers=True)

    logger.info(f"Path where all results are stored: {cfg.output_dir}")

    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)

    logger.info("Building Lightning Module")
    lightning_module = AgentLightningModule(
        agent=agent,
    )

    if cfg.use_cache_without_dataset:
        logger.info("Using cached data without building SceneLoader")
        assert cfg.force_cache_computation==False, "force_cache_computation must be False when using cached data without building SceneLoader"
        assert cfg.cache_path is not None, "cache_path must be provided when using cached data without building SceneLoader"
        train_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.train_logs,
        )
        val_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.val_logs,
        )
    else:
        logger.info("Building SceneLoader")
        train_data, val_data = build_datasets(cfg, agent)

    logger.info("Building Datasets")
    train_dataloader = DataLoader(train_data, **cfg.dataloader.params, shuffle=True)
    logger.info("Num training samples: %d", len(train_data))
    val_dataloader = DataLoader(val_data, **cfg.dataloader.params, shuffle=False)
    logger.info("Num validation samples: %d", len(val_data))

    logger.info("Building Trainer")
    # 复制 Hydra 配置，便于仅在 NPU 分支替换为 Lightning 所需的 Python 对象。
    trainer_params = dict(cfg.trainer.params)
    if trainer_params.get("accelerator") == "npu":
        from navsim.planning.training.npu_accelerator import NPUAccelerator, NPUDDPStrategy
        from pytorch_lightning.strategies import SingleDeviceStrategy

        if trainer_params.get("precision") not in (32, "32", "32-true"):
            raise ValueError("NPU training currently supports only 32-true precision")
        if trainer_params.get("num_nodes", 1) != 1:
            raise ValueError("NPU training currently supports only one node")
        devices = NPUAccelerator.parse_devices(trainer_params.get("devices", "auto"))
        strategy = trainer_params.get("strategy", "auto")
        if strategy not in ("auto", "ddp"):
            raise ValueError("NPU training supports only auto or ddp strategy")
        # 此运行时的 Transformer 融合推理路径会回退到 CPU；关闭以便验证阶段使用 NPU 算子。
        torch.backends.mha.set_fastpath_enabled(False)
        trainer_params["accelerator"] = NPUAccelerator()
        trainer_params["strategy"] = (
            SingleDeviceStrategy(device=torch.device("npu:0"))
            if devices == 1
            else NPUDDPStrategy(process_group_backend="hccl", find_unused_parameters=True)
        )
    # trainer_params['strategy'] = "ddp_find_unused_parameters_true" #TODO
    callbacks = list(agent.get_training_callbacks())
    if cfg.checkpoint_every_n_epochs is not None:
        from pytorch_lightning.callbacks import ModelCheckpoint
        callbacks.append(ModelCheckpoint(
            dirpath=str(Path(cfg.output_dir) / "checkpoints"),
            filename="epoch={epoch:02d}-step={step}",
            auto_insert_metric_name=False,
            every_n_epochs=int(cfg.checkpoint_every_n_epochs),
            save_top_k=-1,
            save_on_train_epoch_end=True,
        ))
    trainer = pl.Trainer(
                **trainer_params, 
                callbacks=callbacks,
                )

    logger.info("Starting Training")
    # ckpt_path = '/data2/yingyan_li/repo/WoTE//exp/training_transfuser_agent/10ep/lightning_logs/version_0/checkpoints/epoch=10-step=3663.ckpt'
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=cfg.resume_from_checkpoint,
    )

if __name__ == "__main__":
    main()
