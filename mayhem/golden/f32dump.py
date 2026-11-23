#!/usr/bin/env python3
# f32dump.py <out.bin>: print funasr-encoder's out.bin (int32 T, int32 D, T*D f32) as text.
import struct, sys
b = open(sys.argv[1], "rb").read()
T, D = struct.unpack("<ii", b[:8])
v = struct.unpack("<%df" % (T * D), b[8:8 + 4 * T * D])
print(T, D)
for t in range(T): print(" ".join("%.5f" % x for x in v[t * D:(t + 1) * D]))
