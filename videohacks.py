import collections
import json
import subprocess
from math import ceil
from subprocess import Popen
from typing import Tuple, Dict, List

import numpy as np

Rational = collections.namedtuple('Rational', ['num', 'den'])


# for [1..255] return 256
# for 256 return 256
# for 0 return 0
def align16up(x: int) -> int:
    return x + 15 & (-15 - 1)


def ffprobejson(path: str):
    process: Popen = subprocess.Popen(
        ['ffprobe', '-hide_banner', '-i', path, '-show_streams', '-show_format', '-print_format', 'json'],
        stdout=subprocess.PIPE)
    return process.communicate()[0]


def ffprobe(path: str):
    return json.loads(ffprobejson(path))


def mediainfo(path: str):
    process: Popen = subprocess.Popen(
        ['mediainfo', path],
        stdout=subprocess.PIPE)
    bytes = process.communicate()[0]
    lines = bytes.split(b'\n')
    kv = [line.split(b':', maxsplit=1) for line in filter(lambda x: b':' in x, lines)]
    d = {key.strip().decode(): value.strip().decode(errors='replace') for (key, value) in kv}
    return d


def video_stream(ffp):
    return next(filter(lambda x: x['codec_type'] == 'video', ffp['streams']))


def parse_rational(r: str) -> Rational:
    (num, den) = r.replace(":", "/").split("/")
    return Rational(int(num), int(den))


def fps(ffp) -> Rational:
    name = ffp['format']['format_name']
    vstream = video_stream(ffp)
    fps: str = vstream['avg_frame_rate'] if name != "mxf" else vstream['r_frame_rate']
    if fps is None:
        return None
    return parse_rational(fps)


def sar(ffp: Dict) -> Rational:
    vstream = video_stream(ffp)
    if vstream is None:
        return None
    sarstr = vstream.get('sample_aspect_ratio', '1/1')
    sar = parse_rational(sarstr)
    return sar


i_dont_care_seek_fast_please_i_know_what_i_am_doing = True


def ffmpeg_cmd(ffp, downscale: float = 2.0, startframe: int = 0, endframe_inclusive: int = -1, interlaced=False,
               crop: str = None, forcesar: Rational = None) -> List[str]:
    vstream = video_stream(ffp)
    bps = int(vstream.get('bits_per_raw_sample', '8'))
    _sar = forcesar if forcesar is not None else sar(ffp)

    if 'nb_frames' in vstream:
        duration: int = int(vstream['nb_frames'])
    else:
        duration: int = 10005000  # infinite
    assert duration > 0, "video has zero frames"
    if endframe_inclusive > -1:
        endframe_inclusive = min(duration - 1, endframe_inclusive)
    duration = min(duration, endframe_inclusive - startframe + 1)
    ss = 0
    if startframe > 0:
        _fps = fps(ffp)
        ss = startframe * _fps.den / _fps.num

    pix_fmt = "bgr48le" if bps > 8 else "bgr24"  # cv2.imread reads images in BGR order
    width = int(vstream['width'])
    height = int(vstream['height'])
    if crop is not None:
        w, h, x, y = crop.split(":")
        width = int(w)
        height = int(h)

    # fix non-square pixel videos
    if _sar.num != 0:
        width = ceil(width * _sar.den / _sar.num)

    width16 = align16up(width)
    height16 = align16up(height)

    if downscale != 1:
        width16 = align16up(int(width / downscale))
        height16 = align16up(int(height / downscale))

    videopath = ffp['format']['filename']

    cmd: List[str] = ['ffmpeg']
    if vstream['codec_name'] == 'prores' or i_dont_care_seek_fast_please_i_know_what_i_am_doing:
        # iframe-only codecs like prores can be quickly and precisely seeked
        if ss > 0:
            cmd.extend(['-ss', str(ss)])
        cmd.extend(['-i', videopath])
    else:
        cmd.extend(['-i', videopath])
        if ss > 0:
            cmd.extend(['-ss', str(ss)])
    if endframe_inclusive > -1:
        cmd.extend(['-vframes', str(duration)])

    filters = [f"format={pix_fmt}"]

    if interlaced:
        filters += ["yadif"]

    if crop is not None:
        filters += [f"crop={crop}"]

    if width != width16 or height != height16 or _sar != (1, 1):
        filters += [f"scale={width16}:{height16}"]

    if len(filters) != 0:
        cmd += ["-vf", ",".join(filters)]

    if width != width16 or height != height16 or _sar != (1, 1):
        cmd.extend(['-sws_flags', 'lanczos'])

    cmd.extend(['-pix_fmt', pix_fmt, '-an', '-sn', '-f', 'rawvideo', '-'])

    return cmd


def videosize(ffp: Dict, adjust_for_sar=True) -> Tuple[int, int]:
    vstream = video_stream(ffp)
    if vstream is None:
        return None

    width = int(vstream['width'])
    height = int(vstream['height'])
    if adjust_for_sar:
        sarstr = vstream.get('sample_aspect_ratio', '1/1')
        sar = parse_rational(sarstr)
        width = ceil(width * sar.den / sar.num)

    return int(width), int(height)


# expects (n, h, w, c) image
def imgs2video(outpath: str, imgs: np.ndarray, fps: int = 12, crf: int = 21, in_pix_fmt: str = 'bgr24',
               out_pix_fmt: str = 'yuv420p'):
    assert len(imgs.shape) == 4, f"expected (N, h, w, c) images got {imgs.shape}"

    n, h, w, c = imgs.shape
    enc = VideoEncoder(outpath, in_w=w, in_h=h, fps=fps, crf=crf, in_pix_fmt=in_pix_fmt, out_pix_fmt=out_pix_fmt)
    for img in imgs:
        enc.imwrite(img)
    enc.close()


class VideoEncoder:
    def __init__(self, outpath: str, in_w: int, in_h: int, fps: float = 24, crf: int = 21, in_pix_fmt: str = 'bgr24',
                 out_pix_fmt: str = 'yuv420p'):
        self.in_h = in_h
        self.in_w = in_w
        self.cmd = ['ffmpeg',
                    '-v', 'info',
                    '-f', 'rawvideo',
                    '-pix_fmt', in_pix_fmt, "-s:v", "%dx%d" % (in_w, in_h),
                    '-r', str(fps),
                    '-i', '-',
                    '-vcodec', 'libx264', '-g', str(int(fps)), '-bf', '0', '-crf', str(crf), '-pix_fmt', out_pix_fmt,
                    '-movflags', 'faststart', '-y', outpath]
        self.process = None

    def forkffmpeg(self):
        print(self.cmd)
        self.process = Popen(self.cmd, stdin=subprocess.PIPE)

    def imwrite(self, img: np.ndarray):
        if self.process is None:
            self.forkffmpeg()
        h, w, c = img.shape
        assert self.in_w == w and self.in_h == h
        uimg = img
        if img.dtype != np.uint8:
            uimg = img.astype(np.uint8)
        self.process.stdin.write(uimg.tobytes('C'))

    def close(self):
        if self.process is not None:
            self.process.stdin.close()
            self.process.wait(10.0)


class SimpleVideoLoader:
    def __init__(self, videopath: str, downscale: float = 1.0, startframe: int = 0,
                 endframe_inclusive: int = -1,
                 crop: str = None,
                 forcesar: Rational = None):
        super(SimpleVideoLoader).__init__()
        assert startframe >= 0, "cant start with negative frame"
        self.videopath = videopath
        self.downscale = downscale
        self.previdx = -1
        self.process: Popen = None
        self.bytes_per_sample: int = 1
        self.nextfn = 0
        ffp = ffprobe(videopath)
        vstream = video_stream(ffp)
        assert vstream is not None, "video stream not found in " + videopath
        if 'nb_frames' in vstream:
            self.duration = int(vstream['nb_frames'])
        else:
            self.duration = 10005000  # infinite

        assert self.duration > 0, "video has zero frames"
        if endframe_inclusive < 0:
            endframe_inclusive = self.duration - 1
        self.duration = min(self.duration, endframe_inclusive - startframe + 1)

        bps = int(vstream.get('bits_per_raw_sample', 8))
        self.bytes_per_sample = 2 if bps > 8 else 1
        self.max_val = 65535. if bps > 8 else 255.
        self.want255 = 1. / 257. if bps > 8 else 1.
        uint16le = np.dtype(np.uint16)
        uint16le = uint16le.newbyteorder('L')
        self.dtype = uint16le if bps > 8 else np.uint8

        width = int(vstream['width'])
        height = int(vstream['height'])
        if crop is not None:
            w, h, x, y = crop.split(":")
            width = int(w)
            height = int(h)

        # fix non-square pixel videos
        _sar = forcesar if forcesar is not None else sar(ffp)
        if _sar.num != 0:
            width = ceil(width * _sar.den / _sar.num)

        self.width16 = align16up(int(width / downscale))
        self.height16 = align16up(int(height / downscale))
        try:
            minfo = mediainfo(videopath)
            interlaced = minfo.get('Scan type', '') != 'Progressive'
        except:
            interlaced = False

        self.cmd = ffmpeg_cmd(ffp, downscale, startframe, endframe_inclusive, interlaced=interlaced, crop=crop,
                              forcesar=forcesar)
        self.fn = 0
        print(" ".join(self.cmd))

    def _forkffmpeg(self):
        self.process: Popen = subprocess.Popen(self.cmd, stdout=subprocess.PIPE)

    def _readnextframe(self) -> np.ndarray:
        if self.nextfn >= self.duration:
            print("done")
            raise StopIteration()
        if self.process is None:
            self._forkffmpeg()
        expected_len = self.width16 * self.height16 * 3 * self.bytes_per_sample
        b = self.process.stdout.read(expected_len)
        assert len(b) == expected_len, f"expected {expected_len} got {len(b)}"
        nb = np.frombuffer(b, dtype=np.dtype(self.dtype), count=self.width16 * self.height16 * 3).reshape(
            (self.height16, self.width16, 3))
        self.nextfn += 1
        return nb * self.want255

    def __iter__(self):
        return self

    def __next__(self):
        return self._readnextframe()

    def __len__(self):
        return self.duration
