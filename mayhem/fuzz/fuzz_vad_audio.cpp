// vad-audio: an untrusted audio file (WAV/MP3/FLAC, any rate/channels) through the same path
// as `llama-funasr-vad -m vad.gguf -a <audio>`: funasr_load_audio_16k_mono (miniaudio decode +
// resample) then funasr_vad_segments (fbank80, LFR, FSMN-VAD graph, E2EVadModel state machine).
// The VAD model is a fixed tiny synthetic GGUF written once to $TMPDIR (read-only afterwards).
#define FUNASR_AUDIO_IMPLEMENTATION
#include "funasr_audio.h"
#include "funasr_vad.h"
#include "synth_models.h"
#include <cstddef>
#include <cstdint>

static std::string g_model, g_input;

extern "C" int LLVMFuzzerInitialize(int*, char***) {
  g_model = synth::scratch_path("vad-model");
  if (!synth::write_vad(g_model.c_str())) abort();
  g_input = synth::scratch_path("audio");
  return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (!synth::write_bytes(g_input, data, size)) abort();
  std::vector<float> wav;
  if (!funasr_load_audio_16k_mono(g_input.c_str(), wav)) return -1;  // not a decodable audio file
  std::vector<std::pair<int, int>> segs;
  funasr_vad_segments(g_model, wav, 30000, segs, 1);
  return 0;
}
