# Load videos into your ML pipelines

```python
with SimpleVideoLoader("testdata/bipbop15_270_stereo.mp4") as loader:
    for frame in loader:
        print(frame.shape)
```

Output
```
(272, 480, 3)
(272, 480, 3)
...
```

# Load HLS

Read first 5 frames from public HLS url

```python
with SimpleVideoLoader("https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8") as loader:
    for frame in itertools.islice(loader, 5):
        print("frame from live video", frame.shape)
```

Output
```
...
frame from live video (720, 1280, 3)
frame from live video (720, 1280, 3)
...
```

# Force FPS

The following will force fps to 5, ffmpeg will drop frames to achieve the fps   
```python
loader = SimpleVideoLoader("testdata/bipbop15_270_stereo.mp4", forcefps=5)
count = 0
for frame in loader:
    print(frame.shape)
    count += 1

print("read", count, "frames")
assert count == 75

```

# Encode video from Numpy Array
```python
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
assert 3 == nb_frames(ffprobe("test_rgb_3_frames.mp4"))
```

# Encode video with precise timestamp control
```python
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
```
```
# ffprobe -show_frames test_rgb_3_frames_precise_pts.mp4 | grep pts
pts=1001
pts=2002
pts=10010
```


# ML pipeline example

```python
with SimpleVideoLoader("testdata/bipbop15_270_stereo.mp4") as loader:
    with VideoEncoder("test_simple.mp4", in_w=480, in_h=270, fps=30) as enc:
        for frame in loader:
            # insert your modifications to the frame here
            modified_frame = frame
            enc.imwrite(modified_frame)
self.assertEqual(449, nb_frames(ffprobe("test_simple.mp4")))
```
