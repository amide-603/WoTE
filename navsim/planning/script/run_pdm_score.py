import pandas as pd
from tqdm import tqdm
import traceback

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from pathlib import Path
from typing import Any, Dict, List, Union, Tuple
from dataclasses import asdict
from datetime import datetime
import logging
import lzma
import pickle
import os
import uuid
import tempfile
import multiprocessing
import re
from concurrent.futures import ProcessPoolExecutor
import numpy as np
# 使用 PyTorch 控制 NPU 上的 Transformer 推理路径。
import torch

from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.utils.multithreading.worker_utils import worker_map

from navsim.planning.script.builders.worker_pool_builder import build_worker
from navsim.common.dataloader import MetricCacheLoader
from navsim.agents.abstract_agent import AbstractAgent
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
    PDMSimulator
)
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.common.dataloader import SceneLoader, SceneFilter
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.common.dataclasses import SensorConfig, Trajectory

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"


def _score_npu_shard(cfg_data, tokens_per_log, device):
    """Run one disjoint shard in a fresh process bound to one logical NPU."""
    cfg_data["inference_device"] = device
    cfg_data["scene_filter"]["log_names"] = list(tokens_per_log)
    cfg_data["scene_filter"]["tokens"] = [token for tokens in tokens_per_log.values() for token in tokens]
    cfg = OmegaConf.create(cfg_data)
    return run_pdm_score([
        {"cfg": cfg, "log_file": name, "tokens": tokens}
        for name, tokens in tokens_per_log.items()
    ])


def _run_multi_npu(cfg, scene_loader, tokens_to_evaluate, devices):
    if len(set(devices)) != len(devices) or any(not re.fullmatch(r"npu:\d+", d) for d in devices):
        raise ValueError("inference_devices must contain unique logical devices such as npu:0")
    if not tokens_to_evaluate:
        raise ValueError("No scenarios have both a scene and a metric cache")
    eligible = set(tokens_to_evaluate)
    token_logs = sorted(
        (token, name)
        for name, tokens in scene_loader.get_tokens_list_per_log().items()
        for token in tokens if token in eligible
    )
    shards = [{} for _ in devices]
    for index, (token, name) in enumerate(token_logs):
        shards[index % len(devices)].setdefault(name, []).append(token)

    # Strip the potentially very large filter lists before copying the config to
    # workers. Each child receives only the scenes assigned to its device.
    cfg_data = OmegaConf.to_container(cfg, resolve=True)
    cfg_data["scene_filter"]["tokens"] = []
    cfg_data["scene_filter"]["log_names"] = []
    tasks = [(shard, device) for shard, device in zip(shards, devices) if shard]
    logger.info("Running %d NPU inference workers for %d scenarios", len(tasks), len(token_logs))
    with ProcessPoolExecutor(max_workers=len(tasks), mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(_score_npu_shard, cfg_data, shard, device) for shard, device in tasks]
        rows = [row for future in futures for row in future.result()]
    observed = [row["token"] for row in rows]
    if len(observed) != len(eligible) or set(observed) != eligible:
        raise RuntimeError("Multi-NPU evaluation did not return each selected scenario exactly once")
    return rows


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    build_logger(cfg)

    # Extract scenes based on scene-loader to know which tokens to distribute across workers
    # TODO: infer the tokens per log from metadata, to not have to load metric cache and scenes here
    scene_loader = SceneLoader(
        sensor_blobs_path=None,
        data_path=Path(cfg.navsim_log_path),
        scene_filter=instantiate(cfg.scene_filter),
        sensor_config=SensorConfig.build_no_sensors(),
    )
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))

    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    num_missing_metric_cache_tokens = len(set(scene_loader.tokens) - set(metric_cache_loader.tokens))
    num_unused_metric_cache_tokens = len(set(metric_cache_loader.tokens) - set(scene_loader.tokens))
    if num_missing_metric_cache_tokens > 0:
        logger.warning(f"Missing metric cache for {num_missing_metric_cache_tokens} tokens. Skipping these tokens.")
    if num_unused_metric_cache_tokens > 0:
        logger.warning(f"Unused metric cache for {num_unused_metric_cache_tokens} tokens. Skipping these tokens.")
    logger.info("Starting pdm scoring of %s scenarios...", str(len(tokens_to_evaluate)))
    data_points = [
        {
            "cfg": cfg,
            "log_file": log_file,
            "tokens": tokens_list,
        }
        for log_file, tokens_list in scene_loader.get_tokens_list_per_log().items()
    ]

    single_eval = getattr(cfg, 'single_eval', False)
    inference_devices = list(cfg.inference_devices)
    if inference_devices and (single_eval or getattr(cfg, "inference_device", None)):
        raise ValueError("inference_devices cannot be combined with single_eval or inference_device")
    # 当前 NPU 适配只验证了单进程单卡，避免多个 worker 同时占用同一设备。
    if str(getattr(cfg, "inference_device", "cpu")).startswith("npu") and not single_eval:
        raise ValueError("NPU inference requires +single_eval=true for a single worker")
    # single-threaded worker_map
    if inference_devices:
        score_rows = _run_multi_npu(cfg, scene_loader, tokens_to_evaluate, inference_devices)
    elif single_eval:
        print("Running single-threaded worker_map")
        score_rows = run_pdm_score(data_points)
    else:
    # mutli-threaded worker_map
        # 仅在多进程评测时创建 worker，单卡推理不启动额外进程。
        worker = build_worker(cfg)
        score_rows: List[Tuple[Dict[str, Any], int, int]] = worker_map(worker, run_pdm_score, data_points)
    

    pdm_score_df = pd.DataFrame(score_rows)
    num_sucessful_scenarios = pdm_score_df["valid"].sum()
    num_failed_scenarios = len(pdm_score_df) - num_sucessful_scenarios
    average_row = pdm_score_df.drop(columns=["token", "valid"]).mean(skipna=True)
    average_row["token"] = "average"
    average_row["valid"] = pdm_score_df["valid"].all()
    pdm_score_df.loc[len(pdm_score_df)] = average_row

    save_path = Path(cfg.output_dir)
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    pdm_score_df.to_csv(save_path / f"{timestamp}.csv")

    logger.info(f"""
        Finished running evaluation.
            Number of successful scenarios: {num_sucessful_scenarios}. 
            Number of failed scenarios: {num_failed_scenarios}.
            Final average score of valid results: {pdm_score_df['score'].mean()}.
            Results are stored in: {save_path / f"{timestamp}.csv"}.
    """)

def run_pdm_score(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[Dict[str, Any]]:
    node_id = int(os.environ.get("NODE_RANK", 0))
    thread_id = str(uuid.uuid4())
    logger.info(f"Starting worker in thread_id={thread_id}, node_id={node_id}")

    log_names = [a["log_file"] for a in args]
    tokens = [t for a in args for t in a["tokens"]]
    cfg: DictConfig = args[0]["cfg"]

    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)
    assert simulator.proposal_sampling == scorer.proposal_sampling, "Simulator and scorer proposal sampling has to be identical"
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()
    agent.is_eval = True
    # 默认沿用 CPU；通过 +inference_device=npu:0 显式启用单卡 NPU。
    device = str(getattr(cfg, "inference_device", "cpu"))
    if device.startswith("npu"):
        import torch_npu  # 注册 PyTorch 的 NPU 设备后端。
        # 当前 NPU 运行时不支持融合的 Transformer 编码器推理算子；
        # 关闭快速路径，使其使用可在 NPU 上执行的常规算子，避免回退 CPU。
        torch.backends.mha.set_fastpath_enabled(False)
        torch.npu.set_device(device)
        logger.info("Using inference device %s", device)
        if cfg.inference_devices:
            print(f"NPU inference worker PID {os.getpid()} using {device} for {len(tokens)} scenarios", flush=True)
    # 将模型和已加载的权重移到指定设备；输入会在 compute_trajectory 中跟随模型。
    agent.to(device)

    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    scene_filter: SceneFilter =instantiate(cfg.scene_filter)
    scene_filter.log_names = log_names
    scene_filter.tokens = tokens
    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    pdm_results: List[Dict[str, Any]] = []
    for idx, (token) in enumerate(tokens_to_evaluate):
        logger.info(
            f"Processing scenario {idx + 1} / {len(tokens_to_evaluate)} in thread_id={thread_id}, node_id={node_id}"
        )
        score_row: Dict[str, Any] = {"token": token, "valid": True}
        try:
            metric_cache_path = metric_cache_loader.metric_cache_paths[token]
            with lzma.open(metric_cache_path, "rb") as f:
                metric_cache: MetricCache = pickle.load(f)
            
            resume_dir = getattr(cfg, "resume_model_results_dir", None)
            saved_result = Path(resume_dir) / f"{token}.npz" if resume_dir else None
            if saved_result is not None and saved_result.is_file():
                with np.load(saved_result, allow_pickle=False) as saved:
                    poses = saved["trajectory"]
                    if poses.shape != (1, 8, 3) or not np.isfinite(poses).all():
                        raise ValueError(f"Invalid saved trajectory: {saved_result}")
                    trajectory = Trajectory(poses.squeeze(0))
            else:
                use_future_frames = agent.config.use_fut_frames if hasattr(agent.config, 'use_fut_frames') else False
                agent_input = scene_loader.get_agent_input_from_token(token, use_fut_frames=use_future_frames)
            if saved_result is not None and saved_result.is_file():
                pass
            elif agent.requires_scene:
                scene = scene_loader.get_scene_from_token(token)
                trajectory = agent.compute_trajectory(agent_input, scene)
            elif cfg.save_model_results and hasattr(agent, "compute_trajectory_with_results"):
                trajectory, model_results = agent.compute_trajectory_with_results(agent_input)
                results_dir = Path(resume_dir) if resume_dir else Path(cfg.output_dir) / "model_results" / str(cfg.experiment_uid)
                results_dir.mkdir(parents=True, exist_ok=True)
                # Write to a temporary file first so interrupted saves are never mistaken for complete results.
                temporary_path = None
                try:
                    with tempfile.NamedTemporaryFile(dir=results_dir, suffix=".npz", delete=False) as temporary:
                        temporary_path = Path(temporary.name)
                        np.savez_compressed(temporary, **model_results)
                    os.replace(temporary_path, results_dir / f"{token}.npz")
                finally:
                    if temporary_path is not None:
                        temporary_path.unlink(missing_ok=True)
            else:
                trajectory = agent.compute_trajectory(agent_input)
            
            pdm_result = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
            )
            score_row.update(asdict(pdm_result))
        except Exception as e:
            logger.warning(f"----------- Agent failed for token {token}:")
            traceback.print_exc()
            score_row["valid"] = False

        pdm_results.append(score_row)
    return pdm_results

if __name__ == "__main__":
    main()
