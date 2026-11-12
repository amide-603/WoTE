"""检查场景并行调度的结果顺序和缓存归属。"""

from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
import unittest
import numpy as np
from navsim.agents.WoTE.score_pool import SceneScorePool


class SceneScorePoolTest(unittest.TestCase):
    def test_cache_affinity_and_order(self):
        # 用轻量替身验证重排后的场景仍回到同一个工作进程并恢复原顺序。
        pool = SceneScorePool.__new__(SceneScorePool)
        pool.processes = [0, 1, 2, 3]
        pool.threads = ThreadPoolExecutor(max_workers=4)
        pool.cache_size = 8
        pool.affinity = OrderedDict()
        pool.last_stats = {}
        calls = []
        def dispatch(worker, tasks):
            calls.append((worker, [task[1] for task in tasks]))
            return ([(index, np.full((5, 2), float(token)), False)
                     for index, token, _path, _proposal in tasks], 100)
        pool._dispatch = dispatch
        tokens = ['0', '1', '2', '3', '4', '5', '6', '7']
        paths = ['dummy'] * len(tokens)
        trajectories = np.zeros((len(tokens), 2, 8, 3))
        try:
            result = pool.score(tokens, paths, trajectories)
            first_assign = {token: worker for worker, group in calls for token in group}
            self.assertEqual([item[0, 0] for item in result], list(map(float, tokens)))
            calls.clear()
            reversed_tokens = list(reversed(tokens))
            result = pool.score(reversed_tokens, paths, trajectories)
            second_assign = {token: worker for worker, group in calls for token in group}
            self.assertEqual(first_assign, second_assign)
            self.assertEqual([item[0, 0] for item in result], list(map(float, reversed_tokens)))
            self.assertEqual(pool.last_stats['requests'], len(tokens))
            with self.assertRaisesRegex(ValueError, '数量必须一致'):
                pool.score(tokens, paths[:-1], trajectories)
        finally:
            pool.threads.shutdown(wait=True)


if __name__ == '__main__':
    unittest.main()
