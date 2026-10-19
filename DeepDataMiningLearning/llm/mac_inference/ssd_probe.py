"""How fast can this SSD feed MoE experts? Random reads of one expert's size, page cache bypassed (F_NOCACHE).

python3 -I mac_inference/ssd_probe.py [model-blobs-dir] [read-size-bytes]
default: the Flash-Next oQ4e shards in the Hugging Face cache, 2.76 MB (one 4-bit expert)
"""
import fcntl, glob, os, random, sys, time
from concurrent.futures import ThreadPoolExecutor

D = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/.cache/huggingface/hub/models--Jundot--Qwen3.8-Flash-Next-oQ4e-mtp/blobs")
SZ = int(sys.argv[2]) if len(sys.argv) > 2 else 2_760_000
F_NOCACHE = 48                                   # macOS fcntl: do not keep these pages in the page cache
fds = []
for f in sorted(glob.glob(os.path.join(D, "*"))):
    if os.path.getsize(f) > 2e9:
        fd = os.open(f, os.O_RDONLY); fcntl.fcntl(fd, F_NOCACHE, 1); fds.append((fd, os.path.getsize(f)))
rng = random.Random(0)


def one(_):
    fd, size = rng.choice(fds)
    off = rng.randrange(0, size - SZ) // 16384 * 16384
    t = time.perf_counter(); os.pread(fd, SZ, off); return time.perf_counter() - t


for threads, n in [(1, 200), (12, 1200), (24, 1200)]:
    t0 = time.perf_counter()
    with ThreadPoolExecutor(threads) as ex:
        lat = sorted(ex.map(one, range(n)))
    dt = time.perf_counter() - t0
    print(f"threads={threads:2d}: {n * SZ / dt / 1e9:5.2f} GB/s, per read p50 {lat[n // 2] * 1e3:5.2f} ms "
          f"p99 {lat[int(n * .99)] * 1e3:5.2f} ms")
