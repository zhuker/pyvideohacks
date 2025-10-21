import unittest

import numpy as np

from videohacks import VideoEncoder, SimpleVideoLoader, ffprobe, nb_frames, video_stream, parse_rational, \
    TimePreciseVideoEncoder, make_nut_header, make_frame_header


def video_frame_pts(ffp):
    vs = video_stream(ffp)
    if "frames" in ffp:
        return [p['pts'] for p in ffp['frames'] if p['stream_index'] == vs['index']]
    elif "packets" in ffp:
        return [p['pts'] for p in ffp['packets'] if p['stream_index'] == vs['index']]
    else:
        raise Exception("No frames or packets found")


class EncoderTestCase(unittest.TestCase):
    def test_rgb_3_frames(self):
        w = 320
        h = 240
        frames = [
            np.frombuffer(b'\x00\x00\xFF' * (w * h), dtype=np.uint8).reshape((h, w, 3)),
            np.frombuffer(b'\x00\xFF\x00' * (w * h), dtype=np.uint8).reshape((h, w, 3)),
            np.frombuffer(b'\xFF\x00\x00' * (w * h), dtype=np.uint8).reshape((h, w, 3)),
        ]
        with VideoEncoder("test_rgb_3_frames.mp4", in_w=w, in_h=h, fps=30) as enc:
            for frame in frames:
                enc.imwrite(frame)
        self.assertEqual(3, nb_frames(ffprobe("test_rgb_3_frames.mp4")))

    def test_rgb_3_frames_precise_pts(self):
        w = 320
        h = 240
        frames = [
            np.frombuffer(b'\x00\x00\xFF' * (w * h), dtype=np.uint8).reshape((h, w, 3)),
            np.frombuffer(b'\x00\xFF\x00' * (w * h), dtype=np.uint8).reshape((h, w, 3)),
            np.frombuffer(b'\xFF\x00\x00' * (w * h), dtype=np.uint8).reshape((h, w, 3)),
        ]
        timestamps = [1001, 2002, 10010]
        with TimePreciseVideoEncoder("test_rgb_3_frames_precise_pts.mp4", timebase=(1, 30000)) as enc:
            for pts, frame in zip(timestamps, frames):
                enc.imwrite(frame, pts=pts)
        ffp = ffprobe("test_rgb_3_frames_precise_pts.mp4", show_frames=True)
        self.assertEqual(3, nb_frames(ffp))
        self.assertEqual(timestamps, video_frame_pts(ffp))

    def test_simple(self):
        with SimpleVideoLoader("testdata/bipbop15_270_stereo.mp4") as loader:
            with VideoEncoder("test_simple.mp4", in_w=480, in_h=270, fps=30) as enc:
                for frame in loader:
                    enc.imwrite(frame)
        self.assertEqual(449, nb_frames(ffprobe("test_simple.mp4")))

    def test_simple_preserve_pts(self):
        ffp = ffprobe("testdata/bipbop15_270_stereo.mp4", show_frames=True)
        vs = video_stream(ffp)
        time_base = parse_rational(vs['time_base'])

        with SimpleVideoLoader("testdata/bipbop15_270_stereo.mp4", metadata=True) as loader:
            with TimePreciseVideoEncoder("test_simple_preserve_pts.mp4", timebase=time_base) as enc:
                for fn, frame, meta in loader:
                    enc.imwrite(frame, pts=meta['pts'])
        ffp_out = ffprobe("test_simple_preserve_pts.mp4", show_frames=True)
        self.assertEqual(449, nb_frames(ffp_out))
        actual_pts = video_frame_pts(ffp_out)
        expected_pts = video_frame_pts(ffp)
        self.assertEqual(expected_pts, actual_pts)

    def test_simple_nut(self):
        w = 320
        h = 240
        colors = [
            b'\x00\x00\xFF',
            b'\x00\xFF\x00',
            b'\xFF\x00\x00',
        ]
        timestamps = [1001, 5005, 10010]

        with open('test_simple_nut.nut', 'wb') as f:
            f.write(make_nut_header((1, 30000), w, h))
            frame_size_bytes = w * h * 3
            for fn, color in enumerate(colors):
                f.write(make_frame_header(timestamps[fn], frame_size_bytes))
                f.write(colors[fn] * (w * h))

        ffp = ffprobe("test_simple_nut.nut", show_packets=True)
        self.assertEqual(timestamps, video_frame_pts(ffp))

    def test_nut_preserve_pts(self):
        ffp = ffprobe("testdata/bipbop15_270_stereo.mp4", show_frames=True)
        vs = video_stream(ffp)
        time_base = parse_rational(vs['time_base'])
        with SimpleVideoLoader("testdata/bipbop15_270_stereo.mp4", metadata=True) as loader:
            with open("test_nut_preserve_pts.nut", "wb") as f:
                f.write(make_nut_header(time_base, vs['width'], vs['height'], fourcc=b"BGR\x18"))
                frame_size_bytes = vs['width'] * vs['height'] * 3
                for fn, frame, meta in loader:
                    pts_ = meta['pts']
                    f.write(make_frame_header(pts_, frame_size_bytes))
                    uimg = frame
                    if frame.dtype != np.uint8:
                        uimg = frame.astype(np.uint8)
                    f.write(uimg.tobytes('C'))
        ffp_nut = ffprobe("test_nut_preserve_pts.nut", show_frames=True)
        expected_pts = video_frame_pts(ffp)
        actual_pts = video_frame_pts(ffp_nut)
        self.assertEqual(expected_pts, actual_pts)


if __name__ == '__main__':
    unittest.main()
