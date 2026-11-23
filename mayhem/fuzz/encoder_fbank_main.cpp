// encoder-fbank: file-input wrapper around upstream's llama-funasr-encoder main()
// (funasr-encoder.cpp compiled with -Dmain=funasr_encoder_main). The input file is the
// untrusted fbank.bin (int32 T, int32 F, T*F float32); the model is a fixed tiny synthetic
// SAN-M encoder GGUF written to $TMPDIR. Equivalent to
//   llama-funasr-encoder -m enc.gguf -f <input> -o <scratch>
// It is a raw (non-libFuzzer) target because upstream exit()s on a missing tensor.
#include "synth_models.h"
int funasr_encoder_main(int argc, char** argv);
int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s fbank.bin\n", argv[0]); return 2; }
  std::string model = synth::scratch_path("enc-model"), out = synth::scratch_path("enc-out");
  if (!synth::write_encoder(model.c_str())) abort();
  std::string a0 = argv[0];
  const char* av[] = {a0.c_str(), "-m", model.c_str(), "-f", argv[1], "-o", out.c_str(), nullptr};
  return funasr_encoder_main(7, (char**)av);
}
