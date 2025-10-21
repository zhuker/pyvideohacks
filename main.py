import itertools

from videohacks import SimpleVideoLoader

loader = SimpleVideoLoader("testdata/bipbop15_270_stereo.mp4", forcefps=5)
count = 0
for frame in loader:
    print(frame.shape)
    count += 1

print("read", count, "frames")
assert count == 75

loader = SimpleVideoLoader("https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8")
for frame in itertools.islice(loader, 5):
    print("frame from live video", frame.shape)

loader = SimpleVideoLoader("testdata/bipbop15_270_stereo.mp4")
count = 0
for frame in loader:
    print(frame.shape)
    count += 1

print("read", count, "frames")
assert count == 449
