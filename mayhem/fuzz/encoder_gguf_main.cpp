// encoder-gguf: file-input wrapper around upstream's llama-funasr-encoder main()
// (funasr-encoder.cpp compiled with -Dmain=funasr_encoder_main). The input file is the
// untrusted encoder GGUF (config keys, tensor table); the fbank.bin is a fixed T=6, F=8
// matrix written to $TMPDIR. Equivalent to
//   llama-funasr-encoder -m <input> -f fbank.bin -o <scratch>
// It is a raw (non-libFuzzer) target because upstream exit()s on a missing tensor.
#include "synth_models.h"
int funasr_encoder_main(int argc, char** argv);
int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s enc.gguf\n", argv[0]); return 2; }
  std::string fb = synth::scratch_path("enc-fbank"), out = synth::scratch_path("enc-out");
  if (!synth::write_fbank(fb.c_str())) abort();
  std::string a0 = argv[0];
  const char* av[] = {a0.c_str(), "-m", argv[1], "-f", fb.c_str(), "-o", out.c_str(), nullptr};
  return funasr_encoder_main(7, (char**)av);
}
