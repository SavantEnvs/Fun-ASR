// vad-gguf: an untrusted FSMN-VAD GGUF model through `funasr_vad_segments(path, wav, ...)`
// (what `llama-funasr-vad -m <gguf> -a fixed.wav` runs): gguf_init_from_file, funasr_vad_load's
// key/tensor validation, then funasr_vad_scores/the state machine with the model's dims.
// The audio is a fixed in-memory 2 s 16 kHz tone burst.
#include "funasr_vad.h"
#include "synth_models.h"
#include <cstddef>
#include <cstdint>

static std::string g_input;
static std::vector<float> g_wav;

extern "C" int LLVMFuzzerInitialize(int*, char***) {
  g_input = synth::scratch_path("vad-gguf");
  g_wav = synth::test_wav();
  return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (!synth::write_bytes(g_input, data, size)) abort();
  std::vector<std::pair<int, int>> segs;
  if (!funasr_vad_segments(g_input, g_wav, 30000, segs, 1)) return -1;  // model rejected
  return 0;
}
