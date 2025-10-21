import collections
import json
import math
import re
import selectors
from math import ceil
from subprocess import DEVNULL, PIPE
from subprocess import Popen
from typing import Tuple, Dict, List

import numpy as np

_DEBUG = False
i_dont_care_seek_fast_please_i_know_what_i_am_doing = True
Rational = collections.namedtuple('Rational', ['num', 'den'])


# for [1..255] return 256
# for 256 return 256
# for 0 return 0
def _align16up(x: int) -> int:
    return x + 15 & (-15 - 1)


def align16up(x: int) -> int:
    return int(x)


def _debug_print(msg):
    if _DEBUG:
        print(msg)


def _popen_stdout(cmd: List[str]) -> Popen:
    _debug_print(" ".join(cmd))
    redirect = None if _DEBUG else DEVNULL
    return Popen(cmd, stdout=PIPE, stderr=redirect, text=False)


def _popen_stdin(cmd: List[str]) -> Popen:
    _debug_print(" ".join(cmd))
    redirect = None if _DEBUG else DEVNULL
    return Popen(cmd, stdin=PIPE, stderr=redirect, stdout=redirect, text=False)


def ffprobejson(path: str, show_packets=False, show_frames=False):
    cmd = ['ffprobe', '-loglevel', 'quiet', '-hide_banner', '-i', path, '-show_streams', '-show_format',
           '-print_format', 'json']
    if show_packets:
        cmd.append('-show_packets')
    if show_frames:
        cmd.append('-show_frames')
    process: Popen = _popen_stdout(cmd)
    return process.communicate()[0]


def ffprobe(path: str, show_packets=False, show_frames=False):
    return json.loads(ffprobejson(path, show_packets, show_frames))


def mediainfo(path: str):
    process: Popen = _popen_stdout(['mediainfo', path])
    bytes = process.communicate()[0]
    lines = bytes.split(b'\n')
    kv = [line.split(b':', maxsplit=1) for line in filter(lambda x: b':' in x, lines)]
    d = {key.strip().decode(): value.strip().decode(errors='replace') for (key, value) in kv}
    return d


def video_stream(ffp):
    return next(filter(lambda x: x['codec_type'] == 'video', ffp['streams']))


def nb_frames(ffp):
    vstream = video_stream(ffp)
    assert vstream is not None, "video stream not found"
    if 'nb_frames' in vstream:
        return int(vstream['nb_frames'])
    if 'packets' in ffp:
        framecount = len([p for p in ffp["packets"] if p["stream_index"] == vstream["index"]])
        return framecount
    return math.nan


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


def ffmpeg_cmd(ffp, downscale: float = 2.0, startframe: int = 0, endframe_inclusive: int = -1, interlaced=False,
               crop: str = None, forcesar: Rational = None, forcefps: float = None,
               forcedimension: Tuple[int, int] = None, forcepixfmt: str = None) -> List[str]:
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
    if forcepixfmt is not None:
        pix_fmt = forcepixfmt
        bps = 8
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
    if forcedimension is not None:
        width16, height16 = forcedimension

    videopath = ffp['format']['filename']

    cmd: List[str] = ['ffmpeg', '-hide_banner']
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

    if forcefps is not None:
        filters.append(f"fps={forcefps}")

    if interlaced:
        filters += ["yadif"]

    if crop is not None:
        filters += [f"crop={crop}"]

    if width != width16 or height != height16 or _sar != (1, 1):
        filters += [f"scale={width16}:{height16}"]

    filters.append("showinfo")
    if len(filters) != 0:
        cmd += ["-vf", ",".join(filters)]

    if width != width16 or height != height16 or _sar != (1, 1):
        cmd.extend(['-sws_flags', 'lanczos'])

    cmd.extend(["-vsync", "0"])
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


class TimePreciseVideoEncoder:
    def __init__(self, outpath: str,
                 fps: float = 30,
                 timebase: Tuple[int, int] = None,
                 codec: str = 'libx264',
                 gop: int = None,
                 crf: int = 21,
                 in_pix_fmt: str = 'bgr24',
                 out_pix_fmt: str = 'yuv420p'):
        self.outpath = outpath
        self.timebase = timebase
        if self.timebase is None:
            if fps.is_integer():
                self.timebase = (1, int(fps))
            else:
                self.timebase = (1000, int(fps * 1000))
        self.codec = codec
        self.crf = crf
        self.in_pix_fmt = in_pix_fmt
        self.out_pix_fmt = out_pix_fmt
        self.gop = gop
        self.process = None
        self.fn = 0
        self.width_ = 0
        self.height_ = 0

    def forkffmpeg(self, w: int, h: int):
        # Set the timescale written in the movie header box (mvhd). Range is 1 to INT_MAX. Default is 1000.
        # only relevant if container is MP4/MOV
        movie_timescale = round(max(1000, self.timebase[1] / self.timebase[0]))
        self.cmd = ['ffmpeg',
                    '-hide_banner',
                    '-v', 'info',
                    '-f', 'nut',
                    '-i', '-',
                    '-vsync', '0',
                    '-movie_timescale', str(movie_timescale),
                    '-copyts', '-enc_time_base', f'{self.timebase[0]}:{self.timebase[1]}'
                    ]
        self.cmd.extend(['-vcodec', self.codec])

        if self.gop is not None:
            self.cmd.extend(['-g', str(self.gop)])

        self.cmd.extend(['-bf', '0', '-crf', str(self.crf),
                         '-pix_fmt', self.out_pix_fmt,
                         # '-movflags', 'faststart',
                         '-y', self.outpath])

        self.process = _popen_stdin(self.cmd)
        fourcc = None
        if self.in_pix_fmt == 'bgr24':
            fourcc = b"BGR\x18"
        elif self.in_pix_fmt == 'rgb24':
            fourcc = b"RGB\x18"
        assert fourcc is not None, f"unsupported pixel format {self.in_pix_fmt}"
        self.width_ = w
        self.height_ = h
        self.process.stdin.write(make_nut_header(self.timebase, width=w, height=h, fourcc=fourcc))

    def imwrite(self, img: np.ndarray, pts_time: float = None, pts: int = None):
        h, w, c = img.shape
        if self.process is None:
            self.forkffmpeg(w, h)
        assert self.width_ == w and self.height_ == h

        if pts is None:
            if pts_time is None:
                pts = self.fn
            else:
                pts = round(pts_time * (self.timebase[1] / self.timebase[0]))
        self.process.stdin.write(make_frame_header(pts, w * h * 3))

        uimg = img
        if img.dtype != np.uint8:
            uimg = img.astype(np.uint8)

        self.process.stdin.write(uimg.tobytes('C'))
        self.fn += 1

    def close(self):
        _close_process(self.process)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return True


class VideoEncoder:
    def __init__(self, outpath: str, in_w: int, in_h: int, fps: float = 24, crf: int = 21, in_pix_fmt: str = 'bgr24',
                 out_pix_fmt: str = 'yuv420p'):
        self.in_h = in_h
        self.in_w = in_w
        self.cmd = ['ffmpeg', '-hide_banner',
                    '-v', 'info',
                    '-f', 'rawvideo',
                    '-pix_fmt', in_pix_fmt, "-s:v", "%dx%d" % (in_w, in_h),
                    '-r', str(fps),
                    '-i', '-',
                    '-vcodec', 'libx264', '-g', str(int(fps)), '-bf', '0', '-crf', str(crf), '-pix_fmt', out_pix_fmt,
                    '-movflags', 'faststart', '-y', outpath]
        self.process = None

    def forkffmpeg(self):
        self.process = _popen_stdin(self.cmd)

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
        _close_process(self.process)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return True


def parse_showinfo(line):
    pattern = re.compile(r'(\w+):\s*(\[.*?\]|\S+)')

    info = dict(pattern.findall(line))

    # Optional: convert numeric-looking values to numbers
    for k, v in info.items():
        if v.startswith('[') and v.endswith(']'):
            info[k] = v.strip('[]')
        elif v.isdigit():
            info[k] = int(v)
        else:
            try:
                info[k] = float(v)
            except ValueError:
                pass
    return info


def _close_process(process, wait: float = None):
    if process is not None:
        _debug_print("closing ffmpeg")
        if process.stdin is not None:
            process.stdin.close()
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()
        if wait is not None:
            process.wait(wait)
        else:
            process.wait()
        _debug_print("ffmpeg closed")


class SimpleVideoLoader:
    def __init__(self, videopath: str, downscale: float = 1.0, startframe: int = 0,
                 endframe_inclusive: int = -1,
                 crop: str = None,
                 forcesar: Rational = None,
                 forcefps: float = None,
                 forcedimension: Tuple[int, int] = None,
                 forcepixfmt: str = None,
                 metadata: bool = False, ):
        super(SimpleVideoLoader).__init__()
        assert startframe >= 0, "cant start with negative frame"
        self.needmetadata = metadata
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
        if forcepixfmt is not None:
            bps = 8
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
        if forcedimension is not None:
            self.width16 = forcedimension[0]
            self.height16 = forcedimension[1]

        try:
            minfo = mediainfo(videopath)
            interlaced = minfo.get('Scan type', '') != 'Progressive'
        except:
            interlaced = False

        self.cmd = ffmpeg_cmd(ffp, downscale, startframe, endframe_inclusive, interlaced=interlaced, crop=crop,
                              forcesar=forcesar, forcefps=forcefps, forcedimension=forcedimension,
                              forcepixfmt=forcepixfmt)
        _debug_print(" ".join(self.cmd))
        self.frame_metadata: Dict[int, Dict] = dict()
        self.frame_metadata_fn = -1
        self.framequeue = []

    def _forkffmpeg(self):
        self.process: Popen = Popen(self.cmd, stdout=PIPE, stderr=PIPE, text=False)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stderr, selectors.EVENT_READ)
        self.selector.register(self.process.stdout, selectors.EVENT_READ)

    def _on_stderr_line(self, line):
        if "Parsed_showinfo" in line:
            info = parse_showinfo(line)
            if "config in" in line:
                self.stream_info = info
            if "n" in info:
                self.frame_metadata_fn = info["n"]
                self.frame_metadata[self.frame_metadata_fn] = info
                if self.stream_info is not None:
                    self.frame_metadata[self.frame_metadata_fn].update(self.stream_info)
            elif self.frame_metadata_fn >= 0 and self.frame_metadata_fn in self.frame_metadata:
                self.frame_metadata[self.frame_metadata_fn].update(info)

    def _take_frame_from_queue(self):
        if len(self.framequeue) > 0:
            fn, nb = self.framequeue[0]
            if fn in self.frame_metadata:
                self.framequeue.pop(0)
                meta = self.frame_metadata[fn]
                toremove = [k for k in self.frame_metadata if k <= fn]
                for k in toremove:
                    self.frame_metadata.pop(k)

                return fn, nb, meta
        return None, None, None

    def _readnextframe(self):
        fn, nb, meta = self._take_frame_from_queue()
        if nb is not None:
            if self.needmetadata:
                return fn, nb, meta
            return nb

        if self.process is None:
            self._forkffmpeg()
            self.framequeue = []

        while self.process.poll() is None:
            # The select() call here blocks until an event is ready (or timeout)
            events = self.selector.select()  # timeout=0.1

            for key, mask in events:
                if key.fileobj == self.process.stderr and mask & selectors.EVENT_READ:
                    line = self.process.stderr.readline()
                    if line:
                        self._on_stderr_line(line.decode().strip())
                elif key.fileobj == self.process.stdout and mask & selectors.EVENT_READ:
                    expected_len = self.width16 * self.height16 * 3 * self.bytes_per_sample

                    b = self.process.stdout.read(expected_len)
                    if len(b) == 0:
                        # end early
                        break
                    assert len(b) == expected_len, f"expected {expected_len} got {len(b)}"
                    nb = np.frombuffer(b, dtype=np.dtype(self.dtype), count=self.width16 * self.height16 * 3).reshape(
                        (self.height16, self.width16, 3))

                    if self.dtype != np.uint8:
                        nb = nb * self.want255
                    self.framequeue.append((self.nextfn, nb))
                    self.nextfn += 1
            fn, nb, meta = self._take_frame_from_queue()
            if nb is not None:
                if self.needmetadata:
                    return fn, nb, meta
                return nb
        _debug_print("process is done")
        for line in self.process.stderr.readlines():
            self._on_stderr_line(line.decode().strip())
        fn, nb, meta = self._take_frame_from_queue()
        if nb is not None:
            if self.needmetadata:
                return fn, nb, meta
            return nb
        raise StopIteration()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _close_process(self.process)
        return True

    def __iter__(self):
        return self

    def __next__(self):
        return self._readnextframe()

    def __len__(self):
        return self.duration


def nut_crc32(data):
    poly = 0x04C11DB7
    crc = 0
    for b in data:
        crc ^= b << 24
        for _ in range(8):
            if crc & 0x80000000:
                crc = ((crc << 1) & 0xFFFFFFFF) ^ poly
            else:
                crc = (crc << 1) & 0xFFFFFFFF
    return crc


def _get_v_len(val: int) -> int:
    return max(1, (val.bit_length() + 6) // 7)


class Buffer:
    def __init__(self):
        self.data = bytearray()

    def write_u(self, val, n):
        for i in range(n - 1, -1, -1):
            self.data.append((val >> (8 * i)) & 0xFF)

    def write_v(self, val):
        if val == 0:
            self.data.append(0)
            return
        bytes_list = []
        while val > 0:
            bytes_list.append(val & 0x7F)
            val >>= 7
        bytes_list.reverse()
        for i in range(len(bytes_list) - 1):
            self.data.append(bytes_list[i] | 0x80)
        if bytes_list:
            self.data.append(bytes_list[-1])

    def write_v_list(self, vals: List[int]):
        for val in vals:
            self.write_v(val)

    def write_s(self, val):
        if val >= 0:
            temp = val * 2
        else:
            temp = -val * 2 - 1
        self.write_v(temp)

    def write_vb(self, data):
        self.write_v(len(data))
        self.data += data


def _write_packet(buffer, startcode, content_data: bytes):
    forward_ptr = len(content_data)
    buffer.write_u(startcode, 8)
    buffer.write_v(forward_ptr + 4)
    if forward_ptr > 4096:
        header_len = 8 + _get_v_len(forward_ptr)
        header_data = buffer.data[-header_len:]
        header_crc = nut_crc32(header_data)
        buffer.write_u(header_crc, 4)
    buffer.data += content_data
    check_data = buffer.data[-forward_ptr:]
    checksum = nut_crc32(check_data)
    buffer.write_u(checksum, 4)


FLAG_KEY = 1  # if set, frame is keyframe
FLAG_CODED_PTS = 8  # ,  // if set, coded_pts is in the frame header
FLAG_SIZE_MSB = 32  # if set, data_size_msb is at frame header, otherwise data_size_msb is 0
FLAG_CHECKSUM = 64  # if set, the frame header contains a checksum
FLAG_INVALID = 8192  # if set, frame_code is invalid


def _write_main_header(content: Buffer, time_base: Tuple[int, int], frame_size_bytes: int):
    # version, stream_count, max_distance, time_base_count, time_base_num, time_base_denom
    content.write_v_list([3, 1, 65536, 1, time_base[0], time_base[1]])

    # frame_code table
    def _write_framecode(flags: int = FLAG_INVALID, size: int = 0, count: int = 1):
        content.write_v_list([flags, 6])  # flags, tmp_fields,
        content.write_s(0)  # tmp_pts
        # tmp_mul, tmp_stream, tmp_size, tmp_res, count
        content.write_v_list([1, 0, size & 0xffff, 0, count])

    # group 1: i=0 to 77 invalid
    _write_framecode(count=78)
    # group 2: skip 'N' and set i=79 good
    _write_framecode(flags=FLAG_KEY | FLAG_CODED_PTS | FLAG_SIZE_MSB | FLAG_CHECKSUM, size=frame_size_bytes, count=1)
    # group 3: i=80 to 255 invalid
    _write_framecode(count=176)

    content.write_v(0)  # header_count_minus1
    return content.data


def _write_stream_header(content, fourcc: bytes, width: int, height: int):
    content.write_v_list([0, 0])  # stream_id, stream_class video
    content.write_vb(fourcc)  # b"RGB\x18"  # fourcc for rgb24
    # time_base_id, msb_pts_shift, max_pts_distance 300*25, decode_delay, stream_flags, codec_specific_data
    # width, height, sample_width, sample_height, colorspace_type
    content.write_v_list([0, 0, 7500, 0, 0, 0, width, height, 1, 1, 0])
    return content.data


def _write_syncpoint(content):
    # pts * tb_count + tb_id,  back_ptr_div16
    content.write_v_list([0 * 1 + 0, 0])
    return content.data


def make_nut_header(time_base: Tuple[int, int], width: int, height: int, fourcc: bytes = b"RGB\x18",
                    frame_size_bytes: int = None):
    buffer = Buffer()
    buffer.data += b'nut/multimedia container\x00'
    frame_size_bytes = width * height * 3 if frame_size_bytes is None else frame_size_bytes

    main_startcode = ((ord('N') << 8) + ord('M')) << 48 | 0x7A561F5F04AD
    _write_packet(buffer, main_startcode, _write_main_header(Buffer(), time_base, frame_size_bytes))

    stream_startcode = ((ord('N') << 8) + ord('S')) << 48 | 0x11405BF2F9DB
    _write_packet(buffer, stream_startcode, _write_stream_header(Buffer(), fourcc, width, height))
    return bytes(buffer.data)


def make_frame_header(pts: int, frame_size_bytes: int) -> bytes:
    buffer = Buffer()
    sync_startcode = ((ord('N') << 8) + ord('K')) << 48 | 0xE4ADEECA4569
    _write_packet(buffer, sync_startcode, _write_syncpoint(Buffer()))
    frame_code = 79
    buffer.write_u(frame_code, 1)
    buffer.write_v(pts + 1)  # coded_pts
    buffer.write_v(frame_size_bytes & 0xffff0000)
    buffer.write_u(0, 4)  # TODO crc checksum should be here but ffmpeg doesnt care
    return bytes(buffer.data)
