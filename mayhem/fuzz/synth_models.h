// synth_models.h — tiny, deterministic synthetic GGUF models + inputs for the Fun-ASR
// runtime tools (FSMN-VAD and the SAN-M encoder), written with ggml's own gguf API.
// The real models are hundreds of MB and downloaded at run time; these use the same key
// and tensor names (export_vad_gguf.py / export_encoder_gguf.py, funasr_vad.h,
// funasr-encoder.cpp) with toy dimensions, so the project code runs end to end.
#pragma once
#include "ggml.h"
#include "gguf.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>
#include <vector>

namespace synth {

struct builder {
  ggml_context* ctx = nullptr;
  gguf_context* gg = nullptr;
  uint32_t rng = 12345u;
  builder(const char* arch) {
    ggml_init_params p = {(size_t)64 * 1024 * 1024, nullptr, false};
    ctx = ggml_init(p);
    gg = gguf_init_empty();
    gguf_set_val_str(gg, "general.architecture", arch);
  }
  ~builder() { gguf_free(gg); ggml_free(ctx); }
  float rnd() {  // deterministic LCG in [-1, 1)
    rng = rng * 1664525u + 1013904223u;
    return ((rng >> 8) & 0xffff) / 32768.0f - 1.0f;
  }
  void u32(const char* k, uint32_t v) { gguf_set_val_u32(gg, k, v); }
  // ne0 = fastest dimension (ggml order). mode: 0 = random*scale, 1 = constant scale
  void t(const std::string& name, int64_t ne0, int64_t ne1, float scale, int mode = 0) {
    ggml_tensor* x = ne1 > 0 ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1)
                             : ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne0);
    ggml_set_name(x, name.c_str());
    float* d = (float*)x->data;
    for (int64_t i = 0; i < ggml_nelements(x); i++) d[i] = mode ? scale : rnd() * scale;
    gguf_add_tensor(gg, x);
  }
  bool write(const char* path) { return gguf_write_to_file(gg, path, false); }
};

// FSMN-VAD: input_dim = lfr_m*80 = 400. The network is a smoothed "energy detector": every
// layer averages its input, so the silence logit falls as the CMVN-normalised log-mel energy
// rises and the state machine sees speech on loud frames and silence on quiet ones.
static bool write_vad(const char* path) {
  const int idim = 400, pd = 6, nl = 2, lorder = 4, od = 5, l1 = 8, l2 = 12;
  builder b("fsmn-vad");
  b.u32("vad.input_dim", idim); b.u32("vad.proj_dim", pd); b.u32("vad.fsmn_layers", nl);
  b.u32("vad.lorder", lorder); b.u32("vad.output_dim", od); b.u32("vad.lfr_m", 5); b.u32("vad.lfr_n", 1);
  b.t("cmvn.shift", idim, 0, -8.0f, 1);
  b.t("cmvn.scale", idim, 0, 0.25f, 1);
  b.t("encoder.in_linear1.linear.weight", idim, l1, 1.0f / idim, 1);
  b.t("encoder.in_linear1.linear.bias", l1, 0, 0.0f, 1);
  b.t("encoder.in_linear2.linear.weight", l1, l2, 1.0f / l1, 1);
  b.t("encoder.in_linear2.linear.bias", l2, 0, 0.0f, 1);
  for (int i = 0; i < nl; i++) {
    std::string p = "encoder.fsmn." + std::to_string(i) + ".";
    b.t(p + "linear.linear.weight", l2, pd, 1.0f / l2, 1);
    b.t(p + "fsmn_block.conv_left.weight", pd, lorder, 0.1f, 1);
    b.t(p + "affine.linear.weight", pd, l2, 1.0f / (pd * 1.4f), 1);
    b.t(p + "affine.linear.bias", l2, 0, 0.0f, 1);
  }
  b.t("encoder.out_linear1.linear.weight", l2, l1, 1.0f / l2, 1);
  b.t("encoder.out_linear1.linear.bias", l1, 0, 0.0f, 1);
  // logit0 (silence) = 2 - 4*energy ; the other 4 logits are 0
  {
    ggml_tensor* x = ggml_new_tensor_2d(b.ctx, GGML_TYPE_F32, l1, od);
    ggml_set_name(x, "encoder.out_linear2.linear.weight");
    float* d = (float*)x->data;
    for (int o = 0; o < od; o++) for (int i = 0; i < l1; i++) d[o * l1 + i] = o == 0 ? -4.0f / l1 : 0.0f;
    gguf_add_tensor(b.gg, x);
    ggml_tensor* bb = ggml_new_tensor_1d(b.ctx, GGML_TYPE_F32, od);
    ggml_set_name(bb, "encoder.out_linear2.linear.bias");
    float* e = (float*)bb->data;
    for (int o = 0; o < od; o++) e[o] = o == 0 ? 2.0f : 0.0f;
    gguf_add_tensor(b.gg, bb);
  }
  return b.write(path);
}

// Compact FSMN-VAD (lfr_m = 1 -> input_dim = 80, one FSMN layer, 2 output classes): the same
// network shape and tensor names as write_vad() in under 4 KiB, so the vad-gguf seed fits inside
// libFuzzer's default max_len (4096) and is not truncated into an unparsable GGUF.
static bool write_vad_small(const char* path) {
  const int idim = 80, pd = 2, nl = 1, lorder = 2, od = 2, l1 = 4, l2 = 4;
  builder b("fsmn-vad");
  b.u32("vad.input_dim", idim); b.u32("vad.proj_dim", pd); b.u32("vad.fsmn_layers", nl);
  b.u32("vad.lorder", lorder); b.u32("vad.output_dim", od); b.u32("vad.lfr_m", 1); b.u32("vad.lfr_n", 1);
  b.t("cmvn.shift", idim, 0, -8.0f, 1);
  b.t("cmvn.scale", idim, 0, 0.25f, 1);
  b.t("encoder.in_linear1.linear.weight", idim, l1, 1.0f / idim, 1);
  b.t("encoder.in_linear1.linear.bias", l1, 0, 0.0f, 1);
  b.t("encoder.in_linear2.linear.weight", l1, l2, 1.0f / l1, 1);
  b.t("encoder.in_linear2.linear.bias", l2, 0, 0.0f, 1);
  for (int i = 0; i < nl; i++) {
    std::string p = "encoder.fsmn." + std::to_string(i) + ".";
    b.t(p + "linear.linear.weight", l2, pd, 1.0f / l2, 1);
    b.t(p + "fsmn_block.conv_left.weight", pd, lorder, 0.1f, 1);
    b.t(p + "affine.linear.weight", pd, l2, 1.0f / (pd * 1.4f), 1);
    b.t(p + "affine.linear.bias", l2, 0, 0.0f, 1);
  }
  b.t("encoder.out_linear1.linear.weight", l2, l1, 1.0f / l2, 1);
  b.t("encoder.out_linear1.linear.bias", l1, 0, 0.0f, 1);
  {
    ggml_tensor* x = ggml_new_tensor_2d(b.ctx, GGML_TYPE_F32, l1, od);
    ggml_set_name(x, "encoder.out_linear2.linear.weight");
    float* d = (float*)x->data;
    for (int o = 0; o < od; o++) for (int i = 0; i < l1; i++) d[o * l1 + i] = o == 0 ? -4.0f / l1 : 0.0f;
    gguf_add_tensor(b.gg, x);
    ggml_tensor* bb = ggml_new_tensor_1d(b.ctx, GGML_TYPE_F32, od);
    ggml_set_name(bb, "encoder.out_linear2.linear.bias");
    float* e = (float*)bb->data;
    for (int o = 0; o < od; o++) e[o] = o == 0 ? 2.0f : 0.0f;
    gguf_add_tensor(b.gg, bb);
  }
  return b.write(path);
}

// SAN-M encoder + adaptor with toy sizes. fbank input dim = ENC_F.
static const int ENC_F = 8;
static bool write_encoder(const char* path) {
  const int F = ENC_F, D = 8, H = 2, FFN = 16, NB = 2, TP = 1, K = 3, LLM = 8, AFFN = 16, AL = 1, AH = 2;
  builder b("funasr-encoder");
  b.u32("funasr.enc.input_size", F); b.u32("funasr.enc.output_size", D);
  b.u32("funasr.enc.attention_heads", H); b.u32("funasr.enc.linear_units", FFN);
  b.u32("funasr.enc.num_blocks", NB); b.u32("funasr.enc.tp_blocks", TP);
  b.u32("funasr.enc.kernel_size", K); b.u32("funasr.adp.llm_dim", LLM);
  b.u32("funasr.adp.ffn_dim", AFFN); b.u32("funasr.adp.n_layer", AL);
  b.u32("funasr.adp.attention_heads", AH);
  auto sanm = [&](const std::string& p, int in) {
    b.t(p + "norm1.weight", in, 0, 1.0f, 1); b.t(p + "norm1.bias", in, 0, 0.1f);
    b.t(p + "self_attn.linear_q_k_v.weight", in, 3 * D, 0.3f); b.t(p + "self_attn.linear_q_k_v.bias", 3 * D, 0, 0.1f);
    b.t(p + "self_attn.fsmn_block.weight", D, K, 0.3f);
    b.t(p + "self_attn.linear_out.weight", D, D, 0.3f); b.t(p + "self_attn.linear_out.bias", D, 0, 0.1f);
    b.t(p + "norm2.weight", D, 0, 1.0f, 1); b.t(p + "norm2.bias", D, 0, 0.1f);
    b.t(p + "feed_forward.w_1.weight", D, FFN, 0.3f); b.t(p + "feed_forward.w_1.bias", FFN, 0, 0.1f);
    b.t(p + "feed_forward.w_2.weight", FFN, D, 0.3f); b.t(p + "feed_forward.w_2.bias", D, 0, 0.1f);
  };
  sanm("audio_encoder.encoders0.0.", F);
  for (int i = 0; i < NB - 1; i++) sanm("audio_encoder.encoders." + std::to_string(i) + ".", D);
  b.t("audio_encoder.after_norm.weight", D, 0, 1.0f, 1); b.t("audio_encoder.after_norm.bias", D, 0, 0.0f, 1);
  for (int i = 0; i < TP; i++) sanm("audio_encoder.tp_encoders." + std::to_string(i) + ".", D);
  b.t("audio_encoder.tp_norm.weight", D, 0, 1.0f, 1); b.t("audio_encoder.tp_norm.bias", D, 0, 0.0f, 1);
  b.t("audio_adaptor.linear1.weight", D, AFFN, 0.3f); b.t("audio_adaptor.linear1.bias", AFFN, 0, 0.1f);
  b.t("audio_adaptor.linear2.weight", AFFN, LLM, 0.3f); b.t("audio_adaptor.linear2.bias", LLM, 0, 0.1f);
  for (int i = 0; i < AL; i++) {
    std::string p = "audio_adaptor.blocks." + std::to_string(i) + ".";
    b.t(p + "norm1.weight", LLM, 0, 1.0f, 1); b.t(p + "norm1.bias", LLM, 0, 0.1f);
    for (const char* n : {"linear_q", "linear_k", "linear_v", "linear_out"}) {
      b.t(p + "self_attn." + n + ".weight", LLM, LLM, 0.3f); b.t(p + "self_attn." + n + ".bias", LLM, 0, 0.1f);
    }
    b.t(p + "norm2.weight", LLM, 0, 1.0f, 1); b.t(p + "norm2.bias", LLM, 0, 0.1f);
    b.t(p + "feed_forward.w_1.weight", LLM, AFFN, 0.3f); b.t(p + "feed_forward.w_1.bias", AFFN, 0, 0.1f);
    b.t(p + "feed_forward.w_2.weight", AFFN, LLM, 0.3f); b.t(p + "feed_forward.w_2.bias", LLM, 0, 0.1f);
  }
  return b.write(path);
}

// fbank.bin as funasr-encoder reads it: int32 T, int32 F, then T*F float32.
static bool write_fbank(const char* path, int T = 6) {
  FILE* f = fopen(path, "wb");
  if (!f) return false;
  int32_t t = T, F = ENC_F;
  fwrite(&t, 4, 1, f); fwrite(&F, 4, 1, f);
  for (int i = 0; i < T * F; i++) { float v = sinf(0.37f * i) * 0.5f; fwrite(&v, 4, 1, f); }
  fclose(f);
  return true;
}

// Fixed 16 kHz mono test signal: 0.3 s silence, 1.2 s tone burst, 0.5 s silence.
static std::vector<float> test_wav() {
  std::vector<float> w(32000, 0.0f);
  for (int i = 4800; i < 24000; i++) w[i] = 0.5f * sinf(2.0f * 3.14159265f * 440.0f * i / 16000.0f);
  return w;
}

// A scratch file under $TMPDIR (fallback /tmp), removed at exit.
static std::vector<std::string>& scratch_list() { static std::vector<std::string> v; return v; }
static void scratch_cleanup() { for (auto& p : scratch_list()) unlink(p.c_str()); }
static std::string scratch_path(const char* tag) {
  const char* d = getenv("TMPDIR");
  std::string tmpl = std::string(d && *d ? d : "/tmp") + "/funasr-" + tag + "-XXXXXX";
  std::vector<char> buf(tmpl.begin(), tmpl.end()); buf.push_back(0);
  int fd = mkstemp(buf.data());
  if (fd < 0) { perror("mkstemp"); abort(); }
  close(fd);
  if (scratch_list().empty()) atexit(scratch_cleanup);
  scratch_list().push_back(buf.data());
  return buf.data();
}
static bool write_bytes(const std::string& path, const uint8_t* data, size_t size) {
  FILE* f = fopen(path.c_str(), "wb");
  if (!f) return false;
  bool ok = size == 0 || fwrite(data, 1, size, f) == size;
  return fclose(f) == 0 && ok;
}

}  // namespace synth
