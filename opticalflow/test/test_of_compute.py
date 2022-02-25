"""System modules"""
import os
import unittest
from opticalflow.of.compute import sparse_v1, sparse_with_statistics_v1


class TestOfCompute(unittest.TestCase):

    ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    RSS_DIR = os.path.join(ROOT_DIR, 'resources')
    TEST_EARTH_VID = os.path.join(RSS_DIR, 'earth.mp4')
    TEST_INVALID_VID = os.path.join(RSS_DIR, 'invalid.mp4')

    # def test_sparse_v1(self):
    #     # Use context manager to test exceptions on invalid video
    #     with self.assertRaises(ConnectionError):
    #         dst_path_1 = os.path.join(self.RSS_DIR, 'invalid_dst.mp4')
    #         sparse_V1(self.TEST_INVALID_VID, dst_path_1)
    #
    #     # Test sparse function on a valid video file earth.mp4
    #     dst_path_2 = os.path.join(self.RSS_DIR, 'earth_dst.mp4')
    #     if os.path.exists(dst_path_2):
    #         os.remove(dst_path_2)
    #     assert os.path.exists(dst_path_2) is False
    #     sparse_V1(self.TEST_EARTH_VID, dst_path_2)
    #     assert os.path.exists(dst_path_2) is True

    def test_sparse_statistics_v1(self):
        # Test sparse function on a valid video file earth.mp4
        dst_path_1 = os.path.join(self.RSS_DIR, 'earth_dst_statistics.mp4')
        if os.path.exists(dst_path_1):
            os.remove(dst_path_1)
        assert os.path.exists(dst_path_1) is False
        sparse_with_statistics_v1(self.TEST_EARTH_VID, dst_path_1)
        assert os.path.exists(dst_path_1) is True

if __name__ == '__main__':
    unittest.main()