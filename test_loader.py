import unittest

from videohacks import SimpleVideoLoader, parse_showinfo


class LoaderTestCase(unittest.TestCase):
    def test_simple(self):
        frames = 0
        with SimpleVideoLoader("testdata/bipbop15_270_stereo.mp4") as loader:
            for frame in loader:
                self.assertEqual((270, 480, 3), frame.shape)
                frames += 1
        self.assertEqual(449, frames)

    def test_simple_forcefps(self):
        frames = 0
        with SimpleVideoLoader("testdata/bipbop15_270_stereo.mp4", forcefps=5) as loader:
            for frame in loader:
                self.assertEqual((270, 480, 3), frame.shape)
                frames += 1
        self.assertEqual(75, frames)

    def test_first_5_frames(self):
        frames = 0
        with SimpleVideoLoader("testdata/bipbop15_270_stereo.mp4", startframe=0, endframe_inclusive=4) as loader:
            for frame in loader:
                self.assertEqual((270, 480, 3), frame.shape)
                frames += 1
        self.assertEqual(5, frames)

    def test_simple_metadata(self):
        frames = 0
        with SimpleVideoLoader("testdata/bipbop15_270_stereo.mp4", metadata=True) as loader:
            for fn, frame, meta in loader:
                self.assertEqual((270, 480, 3), frame.shape)
                frames += 1
        self.assertEqual(449, frames)

    def test_simple_metadata_5_frames(self):
        frames = 0
        with SimpleVideoLoader("testdata/bipbop15_270_stereo.mp4", startframe=0, endframe_inclusive=4,
                               metadata=True) as loader:
            for fn, frame, meta in loader:
                self.assertEqual((270, 480, 3), frame.shape)
                frames += 1
        self.assertEqual(5, frames)

    def test_simple_metadata_forcefps(self):
        frames = 0
        with SimpleVideoLoader("testdata/bipbop15_270_stereo.mp4", forcefps=5, metadata=True) as loader:
            for fn, frame, meta in loader:
                self.assertEqual((270, 480, 3), frame.shape)
                frames += 1
        self.assertEqual(75, frames)

    def test_parse_showinfo(self):
        line = "[Parsed_showinfo_1 @ 0x6000036bc300] n:   0 pts:   2310 pts_time:0.077   duration:   1000 duration_time:0.0333333 fmt:bgr24 cl:unspecified sar:1/1 s:480x270 i:P iskey:1 type:I checksum:9172AA94 plane_checksum:[9172AA94 DD3DF395] mean:[16 2] stdev:[57.5 42.2]"
        expected_info = {'n': 0, 'pts': 2310, 'pts_time': 0.077, 'duration': 1000, 'duration_time': 0.0333333,
                         'fmt': 'bgr24', 'cl': 'unspecified', 'sar': '1/1', 's': '480x270', 'i': 'P', 'iskey': 1,
                         'type': 'I', 'checksum': '9172AA94', 'plane_checksum': '9172AA94 DD3DF395', 'mean': '16 2',
                         'stdev': '57.5 42.2'}
        info = parse_showinfo(line)
        self.assertEqual(expected_info, info)


if __name__ == '__main__':
    unittest.main()
