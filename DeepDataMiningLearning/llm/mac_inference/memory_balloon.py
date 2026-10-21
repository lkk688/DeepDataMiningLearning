"""Make the kernel hand back reclaimable memory on a Jetson, without sudo.

After a CUDA process exits on Jetson (JetPack 7), the driver can keep tens of GB of freed GPU memory pooled.
It is returned under memory pressure or by `echo 3 > /proc/sys/vm/drop_caches` (root). vLLM's start-up check
uses cudaMemGetInfo, which on Jetson reports only MemFree, so it refuses to start while that memory is pooled.

This script creates the pressure: it allocates and touches ordinary memory 1 GiB at a time, stops with a safety
margin when MemAvailable gets low, then exits, which returns everything. Run it only when nothing else needs the memory.

python3 memory_balloon.py [stop_when_available_below_gib=4]
"""
import sys, time

STOP_GIB = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
GIB = 1 << 30


def meminfo(key):
    for line in open("/proc/meminfo"):
        if line.startswith(key + ":"):
            return int(line.split()[1]) * 1024
    raise KeyError(key)


before = meminfo("MemFree")
chunks = []
t0 = time.time()
try:
    while meminfo("MemAvailable") > STOP_GIB * GIB and len(chunks) * GIB < meminfo("MemTotal") - 6 * GIB:
        b = bytearray(GIB)
        b[::4096] = b"\x01" * len(range(0, GIB, 4096))      # touch every page so it is really allocated
        chunks.append(b)
finally:
    n = len(chunks)
    chunks.clear()
time.sleep(2)
print(f"ballooned {n} GiB in {time.time() - t0:.0f}s; MemFree {before / GIB:.1f} -> {meminfo('MemFree') / GIB:.1f} GiB")
