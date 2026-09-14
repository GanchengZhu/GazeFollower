# encoding=utf-8
# Author: GC Zhu
# Email: zhugc2016@gmail.com

import unittest
import time
from gazefollower import GazeFollower, MultiprocessGazeFollower
from gazefollower.misc import DefaultConfig, GazeInfo


class TestMultiprocessing(unittest.TestCase):

    def test_gaze_follower_multiprocessing_instantiation(self):
        cfg = DefaultConfig()
        cfg.use_multiprocessing = True
        gf = GazeFollower(config=cfg)
        self.assertIsInstance(gf, MultiprocessGazeFollower)
        gf.release()

    def test_multiprocess_gaze_follower_lifecycle(self):
        cfg = DefaultConfig()
        mp_gf = MultiprocessGazeFollower(config=cfg)

        # Trigger
        mp_gf.send_trigger(42)

        # Sampling control
        mp_gf.start_sampling()
        time.sleep(0.1)
        info = mp_gf.get_gaze_info()
        # initially or before first inference frame, get_gaze_info is GazeInfo or None
        self.assertTrue(info is None or isinstance(info, GazeInfo))

        mp_gf.stop_sampling()

        # Clean release
        mp_gf.release()
        self.assertFalse(mp_gf._worker_process.is_alive())


if __name__ == '__main__':
    unittest.main()
