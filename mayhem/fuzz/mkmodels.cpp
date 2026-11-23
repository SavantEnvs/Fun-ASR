// funasr_mkmodels <outdir>: writes the synthetic vad.gguf, vad_small.gguf, enc.gguf and fbank.bin used by
// mayhem/test.sh (and as fuzz seeds).
#include "synth_models.h"
int main(int argc, char** argv) {
  if (argc != 2) { fprintf(stderr, "usage: %s outdir\n", argv[0]); return 2; }
  std::string d = argv[1];
  bool ok = synth::write_vad((d + "/vad.gguf").c_str()) && synth::write_vad_small((d + "/vad_small.gguf").c_str()) && synth::write_encoder((d + "/enc.gguf").c_str()) &&
            synth::write_fbank((d + "/fbank.bin").c_str());
  return ok ? 0 : 1;
}
