"""将不同场景分发给独立且常驻的 CPU PDM 评分进程。"""

import atexit
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
import os
import pickle
import subprocess
import sys


class SceneScorePool:
    def __init__(self, workers, fast=None, cache_size=None):
        if workers < 1:
            raise ValueError('评分工作进程数量必须为正数。')
        # 单独启动 CPU 解释器，不从已初始化的 NPU/HCCL 进程 fork Python 状态。
        env = os.environ.copy()
        # 显式配置传递给 CPU 子进程，避免子进程误用训练进程的默认值。
        if fast is not None:
            env['WOTE_SCORE_FAST'] = '1' if fast else '0'
        if cache_size is not None:
            env['WOTE_SCORE_CACHE_SIZE'] = str(cache_size)
        for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            env[name] = '1'
        self.processes = []
        # 启用场景缓存后尽量将同一场景送回同一进程，映射表同样设上限。
        self.cache_size = int(env.get('WOTE_SCORE_CACHE_SIZE', '0'))
        self.affinity = OrderedDict()
        self.last_stats = {}
        self.threads = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='wote-pdm')
        try:
            for _ in range(workers):
                self.processes.append(subprocess.Popen(
                    [sys.executable, '-u', '-m', 'navsim.agents.WoTE.score_worker'],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env,
                ))
        except BaseException:
            self.close()
            raise
        atexit.register(self.close)

    @staticmethod
    def _dispatch(process, tasks):
        # 每个工作进程一次只处理自己的场景，管道结果与进程一一对应。
        try:
            pickle.dump(tasks, process.stdin, protocol=pickle.HIGHEST_PROTOCOL)
            process.stdin.flush()
            ok, result = pickle.load(process.stdout)
        except (EOFError, BrokenPipeError, OSError) as exc:
            raise RuntimeError(f'PDM 评分子进程异常退出，返回码 {process.poll()}。') from exc
        if not ok:
            raise RuntimeError(f'PDM 评分子进程执行失败：\n{result}')
        return result

    def score(self, tokens, paths, trajectories, include_total=False):
        # 一次前向将同批场景均分给进程，按原始索引恢复标签顺序。
        if not (len(tokens) == len(paths) == len(trajectories)):
            raise ValueError('场景 token、缓存路径和候选轨迹的数量必须一致。')
        groups = [[] for _ in self.processes]
        for index, (token, path, proposals) in enumerate(zip(tokens, paths, trajectories)):
            if self.cache_size and token in self.affinity:
                # 命中过的场景按既有归属调度，保留进程内的缓存复用机会。
                worker = self.affinity.pop(token)
                self.affinity[token] = worker
            elif self.cache_size:
                worker = min(range(len(groups)), key=lambda item: len(groups[item]))
                self.affinity[token] = worker
                if len(self.affinity) > self.cache_size * len(groups):
                    self.affinity.popitem(last=False)
            else:
                worker = index % len(groups)
            task = (index, token, str(path), proposals)
            groups[worker].append(task + (True,) if include_total else task)
        futures = [self.threads.submit(self._dispatch, proc, tasks)
                   for proc, tasks in zip(self.processes, groups) if tasks]
        output = [None] * len(tokens)
        hits = 0
        peak_rss_kb = []
        for future in futures:
            values, memory = future.result()
            peak_rss_kb.append(memory)
            for index, scores, cache_hit in values:
                output[index] = scores
                hits += int(cache_hit)
        # 记录本批缓存命中数及进程 RSS 峰值，供性能测试使用。
        self.last_stats = {'hits': hits, 'requests': len(tokens), 'worker_peak_rss_kb': peak_rss_kb}
        return output

    def close(self):
        # 向工作进程发送退出消息并回收资源；异常情况下也不会留下僵尸进程。
        processes, self.processes = self.processes, []
        for process in processes:
            if process.poll() is None:
                try:
                    pickle.dump(None, process.stdin, protocol=pickle.HIGHEST_PROTOCOL)
                    process.stdin.flush()
                except (BrokenPipeError, OSError):
                    pass
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if process.stdin:
                process.stdin.close()
            if process.stdout:
                process.stdout.close()
        self.threads.shutdown(wait=True)
