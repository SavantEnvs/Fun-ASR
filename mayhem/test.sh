#!/usr/bin/env bash
#
# mayhem/test.sh — RUN this repo's OWN functional test suite (already built by mayhem/build.sh).
# exit 0 = pass. EDIT per repo. PATCH-grade oracle: after an agent patches the source, the grader
# rebuilds (build.sh) then runs this. DELETE this file if the repo has no meaningful tests.
#
# IMPORTANT:
#  * Must assert BEHAVIOR/OUTPUT, not just exit status. The oracle has to check asserted values /
#    golden-output diffs / known-answer results — so a PATCH that "fixes" a bug by making the program
#    exit(0) (or any no-op) FAILS here. Running inputs and checking only "exit 0 / didn't crash" is
#    NOT a functional test (it's trivially reward-hackable) — use the project's real assertion suite.
#  * Do NOT build here — mayhem/build.sh already compiled the test suite (with the project's normal
#    flags). This script only RUNS the pre-built tests and reports counts. If the test runner is
#    missing, that's a build.sh bug — fail loudly rather than silently rebuilding.
#  * REQUIRED OUTPUT — a CTRF (https://ctrf.io) summary so Mayhem/the PATCH grader reads the counts:
#      - writes a CTRF JSON report to ${CTRF_REPORT:-$SRC/ctrf-report.json}, and
#      - prints a one-line `CTRF {...}` marker to stdout (same JSON, compact).
#    Only `results.summary` (with tests/passed/failed/pending/skipped/other) is required.
#    Use the emit_ctrf helper below; it computes tests = passed+failed+skipped and sets the exit
#    code (0 iff failed==0). Map your framework's output to passed/failed/skipped.
set -uo pipefail
[ -n "${SOURCE_DATE_EPOCH:-}" ] || unset SOURCE_DATE_EPOCH
: "${MAYHEM_JOBS:=$(nproc)}"   # build parallelism; env-overridable, falls back to nproc (use -j"$MAYHEM_JOBS")
cd "$SRC"

# emit_ctrf <tool> <passed> <failed> [skipped] [pending] [other]
# Writes a CTRF report (file + stdout `CTRF {...}` marker) and returns non-zero iff failed>0.
emit_ctrf() {
  local tool="$1" passed="$2" failed="$3" skipped="${4:-0}" pending="${5:-0}" other="${6:-0}"
  local tests=$(( passed + failed + skipped + pending + other ))
  cat > "${CTRF_REPORT:-$SRC/ctrf-report.json}" <<JSON
{
  "results": {
    "tool": { "name": "$tool" },
    "summary": {
      "tests": $tests,
      "passed": $passed,
      "failed": $failed,
      "pending": $pending,
      "skipped": $skipped,
      "other": $other
    }
  }
}
JSON
  printf 'CTRF {"results":{"tool":{"name":"%s"},"summary":{"tests":%d,"passed":%d,"failed":%d,"pending":%d,"skipped":%d,"other":%d}}}\n' \
    "$tool" "$tests" "$passed" "$failed" "$pending" "$skipped" "$other"
  [ "$failed" -eq 0 ]
}

# Upstream's own regression suite (runtime/llama.cpp/tests/run_regression.sh) needs the real
# FunASR GGUFs downloaded from Hugging Face, so this runs the same tools (built by build.sh with
# upstream's normal Release flags) on tiny deterministic synthetic models and diffs their output
# against committed goldens in mayhem/golden/: VAD segment boundaries on upstream's
# tests/sample.wav (as-is and with silence spliced in, default and VAD_MAXSEG=1500), and the
# encoder's output matrix.
OUT="$SRC/mayhem/out"; G="$SRC/mayhem/golden"
for b in llama-funasr-vad llama-funasr-encoder funasr_mkmodels; do
  [ -x "$OUT/$b" ] || { echo "missing $OUT/$b — run mayhem/build.sh first" >&2; emit_ctrf funasr-golden 0 1; exit 1; }
done
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
pass=0; fail=0
check() { # name, command...
  local name="$1"; shift
  if "$@"; then echo "PASS $name"; pass=$((pass+1)); else echo "FAIL $name"; fail=$((fail+1)); fi
}
"$OUT/funasr_mkmodels" "$W" || true
python3 "$G/pad_wav.py" "$SRC/runtime/llama.cpp/tests/sample.wav" "$W/padded.wav" || true
"$OUT/llama-funasr-vad" -m "$W/vad.gguf" -a "$SRC/runtime/llama.cpp/tests/sample.wav" > "$W/vad_sample.txt" 2>/dev/null
check vad_sample diff -u "$G/vad_sample.txt" "$W/vad_sample.txt"
"$OUT/llama-funasr-vad" -m "$W/vad.gguf" -a "$W/padded.wav" > "$W/vad_padded.txt" 2>/dev/null
check vad_padded diff -u "$G/vad_padded.txt" "$W/vad_padded.txt"
VAD_MAXSEG=1500 "$OUT/llama-funasr-vad" -m "$W/vad.gguf" -a "$W/padded.wav" > "$W/vad_maxseg.txt" 2>/dev/null
check vad_padded_maxseg1500 diff -u "$G/vad_padded_maxseg1500.txt" "$W/vad_maxseg.txt"
"$OUT/llama-funasr-vad" -m "$W/missing.gguf" -a "$W/padded.wav" > "$W/vad_nomodel.txt" 2>&1
check vad_rejects_missing_model grep -q "cannot load" "$W/vad_nomodel.txt"
"$OUT/llama-funasr-encoder" -m "$W/enc.gguf" -f "$W/fbank.bin" -o "$W/enc_out.bin" 2>/dev/null
python3 "$G/f32dump.py" "$W/enc_out.bin" > "$W/enc_out.txt" 2>/dev/null
check encoder_fbank python3 "$G/numcmp.py" "$G/encoder_fbank.txt" "$W/enc_out.txt"
# vad-gguf seeds: every committed seed must load and run the whole VAD pipeline (a seed that fails
# gguf_init_from_file or the tensor check leaves the target fuzzing only the reject path), must be the
# exact model funasr_mkmodels writes, and the compact one must fit libFuzzer's default max_len (4096).
S="$SRC/mayhem/vad-gguf/testsuite"
check vad_gguf_seed_small_fits_maxlen test "$(stat -c %s "$S/vad_small.gguf")" -le 4096
for m in vad vad_small; do
  check "vad_gguf_seed_${m}_matches_generator" cmp "$S/$m.gguf" "$W/$m.gguf"
  "$OUT/llama-funasr-vad" -m "$S/$m.gguf" -a "$W/padded.wav" > "$W/seed_$m.txt" 2> "$W/seed_$m.err"
  check "vad_gguf_seed_${m}_loads" test $? -eq 0 -a -s "$W/seed_$m.txt"
  check "vad_gguf_seed_${m}_no_reject" bash -c '! grep -qE "cannot load|missing" "$1"' _ "$W/seed_$m.err"
done
emit_ctrf funasr-golden "$pass" "$fail"
