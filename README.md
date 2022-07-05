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