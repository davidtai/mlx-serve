#!/bin/bash
# DeepSeek-V4.1 (deepseek_v41: the module-owned native arch over a streamed EXL3 expert bank) host tests: every
# "dsv41 " unit test, on the CPU. Hermetic by default; DSV41_BANK=<bank dir> adds the bank-mode tests (the real
# config, the fill and the admission, the served forward-schedule traces).
#
#   ./tests/test_dsv41.sh
#   DSV41_BANK=~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-exl3-3.0bpw ./tests/test_dsv41.sh
#
# Nothing here loads the model or runs on the GPU: MLX is pinned to the CPU, and the tests that do load it (the
# served cell, the served-schedule reference) run only on explicit window inputs, which the clean environment below
# never passes. The live serving surface (a server and one request) is the bench row: tests/bench.sh --only dsv41.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ZIG="${ZIG:-}"
[ -n "$ZIG" ] || { if [ -x "$ROOT/.zig-toolchain/zig" ]; then ZIG="$ROOT/.zig-toolchain/zig"; else ZIG=zig; fi; }
BIN="$ROOT/zig-out/tests/test"

echo "[build] zig build test-build -Dtest-filter='dsv41 '"
( cd "$ROOT" && "$ZIG" build test-build "-Dtest-filter=dsv41 " --summary none ) || { echo "FAIL: the dsv41 tests do not build"; exit 1; }

# run <label> [VAR=value ...]: the tests in a clean environment (only what they read), one summary line.
run() {
    local label=$1 out rc
    shift
    out=$(env -i HOME="$HOME" PATH="$PATH" TMPDIR="${TMPDIR:-/tmp}" MLX_DEFAULT_DEVICE=cpu "$@" "$BIN" 2>&1)
    rc=$?
    echo "[$label] $(grep -E '^[0-9]+ passed; [0-9]+ skipped; [0-9]+ failed\.$|^All [0-9]+ tests passed\.$' <<< "$out" | tail -1)"
    if [ $rc != 0 ]; then
        grep -E 'FAIL|error' <<< "$out" | head -20
        echo "FAIL: $label (rc $rc)"
        return 1
    fi
}

fails=0
run hermetic || fails=$((fails + 1))
if [ -n "${DSV41_BANK:-}" ]; then
    [ -f "$DSV41_BANK/config.json" ] || { echo "FAIL: $DSV41_BANK/config.json not found"; exit 1; }
    run bank DSV41_BANK="$DSV41_BANK" || fails=$((fails + 1))
else
    echo "[bank] SKIP (DSV41_BANK not set)"
fi
[ $fails = 0 ] && echo "PASS: dsv41 host tests" || { echo "FAIL: $fails run(s)"; exit 1; }
