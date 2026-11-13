import pytorch_lightning as pl
from torch import Tensor
from typing import Dict, Tuple
import torch
from navsim.agents.abstract_agent import AbstractAgent

class AgentLightningModule(pl.LightningModule):
    def __init__(
        self,
        agent: AbstractAgent,
    ):
        super().__init__()
        self.agent = agent

    def _step(
        self,
        batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]],
        logging_prefix: str,
    ):
        features, targets = batch

        input_target = self.agent.config.input_target if hasattr(self.agent.config, 'input_target') else False
        if input_target:
            prediction = self.agent.forward(features, targets)
        else:
            prediction = self.agent.forward(features)

        loss_dict = self.agent.compute_loss(features, targets, prediction)
        if isinstance(loss_dict, Tensor):
            loss_dict = {"traj_loss": loss_dict}
            
        total_loss = 0.0
        for loss_key, loss_value in loss_dict.items():
            self.log(f"{logging_prefix}/{loss_key}", loss_value, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            if 'acc' in loss_key or loss_key.startswith('ranking_component_'):
                continue
            total_loss = total_loss + loss_value
        self.log(f"{logging_prefix}/total_loss", total_loss, on_step=False, on_epoch=True, sync_dist=True)
        return total_loss

    def training_step(
        self,
        batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]],
        batch_idx: int
    ):
        return self._step(batch, "train")

    def validation_step(
        self,
        batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]],
        batch_idx: int
    ):
        return self._step(batch, "val")

    def configure_optimizers(self):
        return self.agent.get_optimizers()

    # 不再重写 backward：Lightning 新版会传入额外参数，使用框架默认实现可兼容训练调用。
