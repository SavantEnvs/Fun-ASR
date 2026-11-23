// vad-stream: the streaming VAD API (funasr_vad_stream_open/feed/close) as
// `llama-funasr-cli --stream` drives it from s16le PCM on stdin. Each input is split by
// FuzzedDataProvider into variable-size PCM chunks (s16le -> f32 exactly like funasr-cli),
// fed in order, then the session is closed. A fresh session (fixed tiny synthetic model)
// is opened per input, so no state crosses inputs.
#include "funasr_vad.h"
#include "synth_models.h"
#include <fuzzer/FuzzedDataProvider.h>
#include <cstddef>
#include <cstdint>

static std::string g_model;

extern "C" int LLVMFuzzerInitialize(int*, char***) {
  g_model = synth::scratch_path("vad-model");
  if (!synth::write_vad(g_model.c_str())) abort();
  return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  FuzzedDataProvider fdp(data, size);
  int max_seg_ms = fdp.ConsumeIntegralInRange<int>(0, 60000);
  funasr_vad_stream* s = funasr_vad_stream_open(g_model.c_str(), max_seg_ms);
  if (!s) abort();  // the fixed model always loads
  s->nthreads = 1;
  std::vector<std::pair<int, int>> segs;
  while (fdp.remaining_bytes() > 0) {
    size_t nsamp = fdp.ConsumeIntegralInRange<size_t>(0, 8000);
    std::vector<uint8_t> raw = fdp.ConsumeBytes<uint8_t>(nsamp * 2);
    std::vector<float> pcm(raw.size() / 2);
    for (size_t i = 0; i < pcm.size(); i++)
      pcm[i] = (int16_t)(raw[2 * i] | (raw[2 * i + 1] << 8)) / 32768.0f;
    if (!funasr_vad_stream_feed(s, pcm.data(), pcm.size(), segs)) break;
  }
  funasr_vad_stream_close(s);
  return 0;
}
