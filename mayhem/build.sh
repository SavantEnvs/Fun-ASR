#!/usr/bin/env bash
#
# mayhem/build.sh — build this repo's fuzz harness(es). EDIT per repo.
#
# Runs inside the commit image (mayhem/Dockerfile) as `mayhem` in /mayhem. The base image
# (ghcr.io/savantenvs/base) already exports the build contract — use these, don't redefine:
#   CC, CXX             stock clang / clang++
#   LIB_FUZZING_ENGINE  -fsanitize=fuzzer   (link into each harness that has a LLVMFuzzer entry)
#   SANITIZER_FLAGS     -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer
#                       (ASan + UBSan, both set to HALT — so Mayhem catches memory AND UB defects)
#   DEBUG_FLAGS         -g -gdwarf-3   (DWARF debug info — always on for fuzz/standalone builds,
#                       independent of the sanitizer off-switch; DWARF version must be < 4)
#   RUST_DEBUG_FLAGS    -C debuginfo=2 -C force-frame-pointers=yes -C llvm-args=-gdwarf-3
#                       (thread through RUSTFLAGS on every cargo-fuzz build)
#   GO_DEBUG_FLAGS      -gcflags=all=-N -l
#                       (thread through go build / go-fuzz-build so the linked ELF keeps symbols)
#   SRC                 /mayhem (the repo source)
#
# Contract: build EVERYTHING here — one runnable binary per fuzz harness, AND the project's test
# suite (so mayhem/test.sh only has to RUN it, never compile). Keep it ADDITIVE (build upstream as
# upstream documents; don't edit upstream files). IMPORTANT: build the PROJECT ITSELF with
# $SANITIZER_FLAGS and $DEBUG_FLAGS (not just the harness) so the fuzzed code is instrumented
# AND carries DWARF < 4 symbols — otherwise ASan/UBSan only see the harness, not the library
# you're trying to find bugs in, and backtraces won't resolve project source lines. Build the TEST
# suite with the project's NORMAL flags (a clean, independent build) so test.sh stays an honest
# functional oracle and won't false-fail on benign UB. Leave the test binary/runner where test.sh
# expects it.
set -euo pipefail

# clang rejects SOURCE_DATE_EPOCH='' (empty) — it must be unset or a valid integer.
[ -n "${SOURCE_DATE_EPOCH:-}" ] || unset SOURCE_DATE_EPOCH

# Build knobs come from the ENVIRONMENT (overridable), with sane defaults — no if-statements,
# just parameter-expansion fallbacks. Default sanitizers are the base's ASan+UBSan (halting); override
# per build via the Dockerfile's `--build-arg SANITIZER_FLAGS="..."`.
# NB: SANITIZER_FLAGS uses `=` (no colon) on purpose — `=` only fills when the var is UNSET, so an
# explicit EMPTY value (`--build-arg SANITIZER_FLAGS=`) is honored and builds with NO sanitizers
# (useful when you want the program's natural crash / full backtrace, not an ASan report). The other
# knobs use `:=` (default on empty too). MAYHEM_JOBS sets build parallelism (falls back to nproc).
: "${SANITIZER_FLAGS=-fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer}"
# DEBUG_FLAGS carries DWARF debug info INDEPENDENTLY of the sanitizer off-switch (so an empty
# SANITIZER_FLAGS still yields DWARF symbols). DWARF MUST be < 4 (Mayhem triage can't read >=4); clang-19's
# plain `-g` emits DWARF-5, so `-gdwarf-3` is explicit. Apply $DEBUG_FLAGS to the fuzz/harness/standalone
# builds (NOT the test/oracle build). Rust/Go carry the same intent via their language flags below.
: "${DEBUG_FLAGS:=-g -gdwarf-3}"
: "${RUST_DEBUG_FLAGS:=-C debuginfo=2 -C force-frame-pointers=yes -C llvm-args=-gdwarf-3}"
: "${GO_DEBUG_FLAGS:=-gcflags=all=-N -l}"
: "${CC:=clang}" ; : "${CXX:=clang++}" ; : "${LIB_FUZZING_ENGINE:=-fsanitize=fuzzer}"
: "${MAYHEM_JOBS:=$(nproc)}"
# COVERAGE_FLAGS: empty by default → no effect on the normal oracle build. Set it via the Dockerfile's
# `--build-arg COVERAGE_FLAGS="-fprofile-instr-generate -fcoverage-mapping"` to instrument the TEST
# build for source-coverage measurement (how much of the project the test suite actually exercises —
# a quality signal for the oracle; complements the anti-reward-hack sabotage check). APPEND it to the
# test build's compile+link flags in step 3 (NOT the fuzz build); after `test.sh` runs, merge with
# `llvm-profdata` and report with `llvm-cov`. Empty value is honored (`=`, not `:=`).
: "${COVERAGE_FLAGS=}"
export SANITIZER_FLAGS DEBUG_FLAGS RUST_DEBUG_FLAGS GO_DEBUG_FLAGS CC CXX LIB_FUZZING_ENGINE MAYHEM_JOBS COVERAGE_FLAGS

cd "$SRC"

# ggml (llama.cpp @ the commit upstream's CMakeLists.txt pins) is prebuilt in the image, static +
# uninstrumented, under /opt/toolchains/deps (mayhem/Dockerfile). No network here.
: "${FUNASR_GGML_PREFIX:=/opt/toolchains/deps/llama.cpp-8086439a4cea94c71a5dfb8fe4ad1546aebd640f}"
GLIB="$FUNASR_GGML_PREFIX/lib"; [ -d "$GLIB" ] || GLIB="$FUNASR_GGML_PREFIX/lib64"
GGML_LIBS="$GLIB/libggml.a $GLIB/libggml-cpu.a $GLIB/libggml-base.a -lpthread -lm"
RT="$SRC/runtime/llama.cpp"
INC="-I$FUNASR_GGML_PREFIX/include -I$RT/funasr-common -I$SRC/mayhem/fuzz"
OUT="$SRC/mayhem/out"            # gitignored (mayhem/.gitignore)
mkdir -p "$OUT/obj"

# 1) Oracle build: upstream's two ggml-only tools exactly as upstream's CMake builds them
#    (C++17, Release -O3 -DNDEBUG), NO sanitizers, NO -gdwarf-3, plus the synthetic-model writer.
CXXSTD="-std=c++17"
$CXX $CXXSTD -O3 -DNDEBUG $COVERAGE_FLAGS $INC "$RT/funasr-vad/funasr-vad.cpp" $GGML_LIBS -o "$OUT/llama-funasr-vad"
$CXX $CXXSTD -O3 -DNDEBUG $COVERAGE_FLAGS $INC "$RT/funasr-encoder/funasr-encoder.cpp" $GGML_LIBS -o "$OUT/llama-funasr-encoder"
$CXX $CXXSTD -O2 $INC "$SRC/mayhem/fuzz/mkmodels.cpp" $GGML_LIBS -o "$OUT/funasr_mkmodels"

# 2) Fuzz build: the project code (header-only funasr_vad.h / funasr_audio.h, funasr-encoder.cpp)
#    is compiled INTO each harness TU with the sanitizers + SanCov (fuzzer-no-link) + DWARF 3.
FZ="$SANITIZER_FLAGS $DEBUG_FLAGS -fsanitize=fuzzer-no-link -O1"
$CXX $CXXSTD $SANITIZER_FLAGS $DEBUG_FLAGS -c "$SRC/mayhem/lsan_off.cc" -o "$OUT/obj/lsan_off.o"
$CC $SANITIZER_FLAGS $DEBUG_FLAGS -c "$STANDALONE_FUZZ_MAIN" -o "$OUT/obj/standalone_main.o"
# ggml's GGUF parser (gguf.cpp) IS the attack surface of vad-gguf / encoder-gguf (an untrusted model
# file), so it is compiled from the pinned dependency source with the same sanitizers + SanCov and
# linked AHEAD of libggml-base.a: the linker then never pulls the archive's uninstrumented gguf.cpp.o.
# The rest of ggml (tensor ops, CPU backend) stays the prebuilt uninstrumented dependency.
GGSRC="$FUNASR_GGML_PREFIX/src/ggml"
$CXX $CXXSTD $FZ -I"$GGSRC/include" -I"$GGSRC/src" -c "$GGSRC/src/gguf.cpp" -o "$OUT/obj/gguf_fuzz.o"
GGUF_FUZZ="$OUT/obj/gguf_fuzz.o"
for h in vad_audio vad_gguf vad_stream; do
  $CXX $CXXSTD $FZ $INC -c "$SRC/mayhem/fuzz/fuzz_$h.cpp" -o "$OUT/obj/fuzz_$h.o"
  $CXX $CXXSTD $SANITIZER_FLAGS $DEBUG_FLAGS $LIB_FUZZING_ENGINE "$OUT/obj/fuzz_$h.o" "$OUT/obj/lsan_off.o" \
      $GGUF_FUZZ $GGML_LIBS -o "$OUT/fuzz_$h"
  $CXX $CXXSTD $SANITIZER_FLAGS $DEBUG_FLAGS "$OUT/obj/standalone_main.o" "$OUT/obj/fuzz_$h.o" \
      "$OUT/obj/lsan_off.o" $GGUF_FUZZ $GGML_LIBS -o "$OUT/fuzz_$h-standalone"
done
# upstream's encoder main(), renamed so the file-input wrappers can supply the fixed half of argv
$CXX $CXXSTD $FZ $INC -Dmain=funasr_encoder_main -c "$RT/funasr-encoder/funasr-encoder.cpp" \
    -o "$OUT/obj/funasr-encoder.o"
for h in fbank gguf; do
  $CXX $CXXSTD $FZ $INC -c "$SRC/mayhem/fuzz/encoder_${h}_main.cpp" -o "$OUT/obj/encoder_${h}_main.o"
  $CXX $CXXSTD $SANITIZER_FLAGS $DEBUG_FLAGS -fsanitize=fuzzer-no-link "$OUT/obj/encoder_${h}_main.o" \
      "$OUT/obj/funasr-encoder.o" "$OUT/obj/lsan_off.o" $GGUF_FUZZ $GGML_LIBS -o "$OUT/encoder_$h"
done
ls -l "$OUT"
