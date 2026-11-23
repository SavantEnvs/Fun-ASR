#!/usr/bin/env python3
# pad_wav.py <in.wav> <out.wav>: 2 s of digital silence, first half of <in>, 2 s silence,
# second half, 2 s silence — gives the VAD state machine real silence/speech transitions.
import sys, wave
r = wave.open(sys.argv[1]); p = r.getparams(); d = r.readframes(p.nframes); r.close()
z = b"\0" * (p.sampwidth * p.nchannels * p.framerate * 2)
h = (len(d) // (2 * p.sampwidth * p.nchannels)) * p.sampwidth * p.nchannels
w = wave.open(sys.argv[2], "wb"); w.setparams(p); w.writeframes(z + d[:h] + z + d[h:] + z); w.close()
