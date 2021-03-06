# Load videos into your ML pipelines

```
loader = SimpleVideoLoader("testdata/bipbop15_270_stereo.mp4")
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

```
loader = SimpleVideoLoader("https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8")
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
```
loader = SimpleVideoLoader("testdata/bipbop15_270_stereo.mp4", forcefps=5)
count = 0
for frame in loader:
    print(frame.shape)
    count += 1

print("read", count, "frames")
assert count == 77

```