#!/usr/bin/env python3
# numcmp.py <golden> <actual>: same shape, every number within 1e-3.
import sys
g = open(sys.argv[1]).read().split(); a = open(sys.argv[2]).read().split()
ok = len(g) == len(a) and len(g) > 0 and all(abs(float(x) - float(y)) <= 1e-3 for x, y in zip(g, a))
sys.exit(0 if ok else 1)
