"""仅运行 CPU PDM 评分的独立子进程，二进制协议不依赖 NPU 上下文。"""

import lzma
import os
import pickle
import resource
import sys
import traceback
from collections import OrderedDict


def main():
    # 将第三方库可能输出的日志与进程间二进制协议隔离。
    output = sys.stdout.buffer
    sys.stdout = sys.stderr
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from navsim.agents.WoTE.pdm_scoring import score_candidates_with_reference

    # 每个进程拥有独立可变模拟器和评分器，避免场景之间产生状态竞争。
    sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    simulator = PDMSimulator(sampling)
    scorer = PDMScorer(sampling, PDMScorerConfig(progress_distance_threshold=5.0))
    # 最终优化可开启有界场景缓存和批量轨迹转换；默认关闭以便分别测速。
    cache_limit = int(os.environ.get('WOTE_SCORE_CACHE_SIZE', '0'))
    fast_trajectory = os.environ.get('WOTE_SCORE_FAST', '0') == '1'
    if cache_limit < 0:
        raise ValueError('WOTE_SCORE_CACHE_SIZE 不能为负数。')
    scene_cache = OrderedDict()
    source = sys.stdin.buffer
    while True:
        try:
            tasks = pickle.load(source)
        except EOFError:
            return
        if tasks is None:
            return
        try:
            values = []
            for index, token, path, proposals in tasks:
                # 评分缓存对象及参考轨迹仅由当前进程使用，按最近使用顺序淘汰。
                key = str(path)
                cache_hit = key in scene_cache
                if cache_hit:
                    cache, reference = scene_cache.pop(key)
                    scene_cache[key] = (cache, reference)
                else:
                    with lzma.open(path, 'rb') as file:
                        cache = pickle.load(file)
                    reference = None
                    if cache_limit:
                        from navsim.evaluate.pdm_score import get_trajectory_as_array
                        reference = get_trajectory_as_array(cache.trajectory, sampling, cache.ego_state.time_point)
                        scene_cache[key] = (cache, reference)
                        if len(scene_cache) > cache_limit:
                            scene_cache.popitem(last=False)
                scores = score_candidates_with_reference(cache, proposals, sampling, simulator, scorer,
                                                          reference_states=reference,
                                                          fast_trajectory=fast_trajectory)
                values.append((index, scores, cache_hit))
            # 峰值 RSS 供测速报告估计各进程内存，不作为训练标签。
            peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            pickle.dump((True, (values, peak_rss_kb)), output, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            # 将场景评分堆栈传回训练进程，避免静默使用缺失标签。
            pickle.dump((False, traceback.format_exc()), output, protocol=pickle.HIGHEST_PROTOCOL)
        output.flush()


if __name__ == '__main__':
    main()
