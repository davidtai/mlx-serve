#!/bin/bash
# bench.sh — the performance bench. llmprobe measures; this drives mlx-serve.
#
# One `llmprobe --bench-only` run per model gives the decode/prefill/TTFT
# medians AND the context ladder. The numbers go into benchmarks.md by hand —
# there is no CSV, no chart pipeline and no engine matrix here any more.
#
#   ./tests/bench.sh                                # every model
#   ./tests/bench.sh --only qwen38-27b              # one row
#   ./tests/bench.sh --url 127.0.0.1:1234 -m <id>   # a server someone else started
#   ./tests/bench.sh --full                         # median of 3 per rung, to 64k
#   BENCH_EXTRA_FLAGS="--wired-margin 2000000000" ./tests/bench.sh --only dsv41   # flags appended to every boot
#
# Each cell is mlx-serve at its FASTEST: speculation is forced on where the
# checkpoint carries an MTP head (it is default-off on MoE targets). The mode
# that actually engaged is printed beside the number, from the server's own
# log — a mode that silently stops engaging shows up as a bare cell.
#
# Comparing against another engine: start it yourself (LM Studio, oMLX, MTPLX,
# llama-server, whatever), then point --url at it. Same protocol, same probe,
# one less thing in this script to keep in sync.
#
# Requirements: node (npx), curl, mlx-serve built ReleaseFast (Debug is 2-4x
# slower = a fake regression).
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ONLY=""
FULL=0
URL=""
URL_MODEL=""
TAG="$(date +%Y%m%d-%H%M%S)"
SETTLE="${SETTLE:-20}"

BINARY="${BINARY:-$ROOT/zig-out/bin/mlx-serve}"
LLMPROBE="${LLMPROBE:-npx -y llmprobe@latest}"
PORT=11250

usage() { sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --only)    ONLY="$2"; shift 2 ;;
        --url)     URL="$2"; shift 2 ;;
        -m|--model) URL_MODEL="$2"; shift 2 ;;
        --full)    FULL=1; shift ;;
        --tag)     TAG="$2"; shift 2 ;;
        --settle)  SETTLE="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown flag: $1 (try --help)" >&2; exit 1 ;;
    esac
done

# Reports live outside the repo: ~/claude-tmp survives reboots, /tmp does not.
OUT="$HOME/claude-tmp/bench-$TAG"
mkdir -p "$OUT"

# ── Model matrix: logical|candidates relative to a model root ──
# The first candidate found wins (tests/_lib_models.sh). A row with no checkpoint
# on this box, or one past its GPU budget, skips: a bench you can't run here
# isn't an error on the box that can.
# ANE=1 adds --ane-prefill to every boot
# (a named refusal on non-qwen3_5-dense models, so it is safe matrix-wide);
# ane-on cells are their own column, never diffed against ane-off ones.
source "$SCRIPT_DIR/_lib_models.sh"
TARGETS=(
    "gemma4-e4b-4bit|mlx-community/gemma-4-e4b-it-4bit"
    "gemma4-26b-a4b-moe-qat-4bit|mlx-community/gemma-4-26B-A4B-it-qat-4bit"
    "qwen36-35b-a3b|ddalcu/Qwen3.6-35B-A3B-MLX-Serve-4bit"
    "qwen38-27b|ddalcu/Qwen3.8-27B-MLX-Serve-4bit"
    "qwen38-27b-iq|ddalcu/Qwen3.8-27B-MLX-Serve-iQ-MLX-3.8bpw"
    "qwen38-flash-next|ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit"
    # DeepSeek-V4.1 over a streamed EXL3 expert bank: sized by the load bill it is admitted on (model_load_gb), one
    # request per server start. On the dev box it only runs as a chain step inside a guarded window
    # (MLX_SERVE_MODEL_ROOTS=~/models ./tests/bench.sh --only dsv41); anywhere else it runs as any row.
    "dsv41-flash-exl3|DeepSeek-V4.1-Flash-MTPLX-streaming-exl3-3.0bpw"
)
# Rows whose server serves one request per start (its phase change is per start): one timed request, not the ladder.
ONE_REQUEST_ROWS=" dsv41-flash-exl3 "
ONE_REQUEST_PROMPT_LEN="${ONE_REQUEST_PROMPT_LEN:-16384}"

# Only ever called on the path that STARTED a server: --url may be pointed at
# a local mlx-serve someone else is using, and a bench must not kill it.
stop_server() {
    pkill -f "mlx-serve --serve .*--port $PORT" 2>/dev/null
    for _ in $(seq 1 30); do
        lsof -ti tcp:"$PORT" >/dev/null 2>&1 || return 0
        sleep 1
    done
}

probe() { # logical host model_id
    local depth=(--bench-only)
    [[ "$FULL" -eq 1 ]] && depth+=(--full)
    echo "── $1 ($2, $3) ──"
    # shellcheck disable=SC2086
    $LLMPROBE "$2" -m "$3" "${depth[@]}" --save "$OUT/$1.json" \
        || echo "  llmprobe failed for $1" >&2
}

# one_request logical port log: a prompt of ONE_REQUEST_PROMPT_LEN ids (the server's own tokenizer), 128 greedy ids,
# the numbers from the server's own timing line; saved in llmprobe's bench shape so the table below reads it.
one_request() {
    printf '── %s (localhost:%s, one request, a %s-id prompt) ──\n' "$1" "$2" "$ONE_REQUEST_PROMPT_LEN"
    python3 - "$2" "$ONE_REQUEST_PROMPT_LEN" "$3" "$OUT/$1.json" <<'PY' || echo "  the request failed for $1" >&2
import json, re, sys, urllib.request
port, n, log, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
def post(path, body, timeout=3600):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())
unit = "The quick brown fox jumps over the lazy dog while the bench counts every id it reads. "
ids = post("/tokenize", {"content": unit * (n // 8 + 64)})["tokens"][:n]
text = post("/detokenize", {"tokens": ids})["content"]
post("/v1/completions", {"model": "mlx-serve", "prompt": text, "max_tokens": 128, "temperature": 0, "ignore_eos": True})
lines = open(log, errors="replace").read()
t = re.findall(r"<- (\d+)\+(\d+) tokens \((\d+)ms\) \[prefill: ([\d.]+) tok/s, decode: ([\d.]+) tok/s\]", lines)
if not t: sys.exit("no timing line in the server log")
prefill, decode = t[-1][3], t[-1][4]
stats = re.findall(r"\[spec-stats\] mode=\w+ attempts=\d+ accepts=\d+ avg_per_round=([\d.]+)", lines)
bench = {"decodeTokPerSec": {"median": float(decode)}, "prefillTokPerSec": {"median": float(prefill)}}
if stats: bench["speculative"] = {"tokensPerStep": float(stats[-1])}
json.dump({"bench": bench, "oneRequest": {"promptIds": len(ids), "maxNew": 128}}, open(out, "w"))
PY
}

# --mtp is forced wherever the checkpoint ships a head: it is default-OFF on
# MoE targets, which is exactly where it pays most (35B-A3B reads 157 without
# and 191 with). On a dense MTP checkpoint it restates the default.
spec_flags() { # model_path
    local f=""
    if ls "$1"/*mtp*.safetensors >/dev/null 2>&1 || [ -d "$1/mtp" ] \
       || grep -qi '"mtp' "$1/config.json" 2>/dev/null; then
        f=" --mtp"
    fi
    [[ "${ANE:-0}" == "1" ]] && f+=" --ane-prefill"
    echo "$f"
}

# ── Run ──
if [[ -n "$URL" ]]; then
    [[ -n "$URL_MODEL" ]] || { echo "--url needs -m <model id>" >&2; exit 1; }
    echo "=== bench: $URL ($URL_MODEL) ==="
    probe "$(echo "$URL_MODEL" | tr '/ ' '__')" "$URL" "$URL_MODEL"
else
    [[ -x "$BINARY" ]] || { echo "no $BINARY — build ReleaseFast first" >&2; exit 1; }
    echo "=== bench: mlx-serve, tag=$TAG, reports → $OUT ==="
    trap 'stop_server' EXIT
    stop_server
    for row in "${TARGETS[@]}"; do
        IFS='|' read -r logical rest <<< "$row"
        [[ -n "$ONLY" && "$logical" != *"$ONLY"* ]] && continue
        IFS='|' read -r -a cands <<< "$rest"
        path=$(find_fitting_model "${cands[@]}") || { echo "SKIP $logical (no checkpoint within $(max_model_gb) GB on this box)" >&2; continue; }
        flags="$(spec_flags "$path")"
        echo; echo ">> $logical$flags"
        # shellcheck disable=SC2086
        "$BINARY" --serve --model "$path" --port "$PORT" $flags ${BENCH_EXTRA_FLAGS:-} >"$OUT/$logical.log" 2>&1 &
        pid=$!
        for _ in $(seq 1 300); do
            curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
            sleep 1
        done
        if ! curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            echo "  mlx-serve never came up for $logical" >&2
        elif [[ "$ONE_REQUEST_ROWS" == *" $logical "* ]]; then
            one_request "$logical" "$PORT" "$OUT/$logical.log"
        else
            probe "$logical" "localhost:$PORT" "$(basename "$path")"
        fi
        kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
        stop_server
        sleep "$SETTLE"
    done
    stop_server
fi

# ── The only artifact: rows to paste into benchmarks.md ──
echo
python3 - "$OUT" <<'PY'
import json, re, sys
from pathlib import Path

for path in sorted(Path(sys.argv[1]).glob("*.json")):
    bench = (json.loads(path.read_text()) or {}).get("bench") or {}
    decode = (bench.get("decodeTokPerSec") or {}).get("median")
    prefill = (bench.get("prefillTokPerSec") or {}).get("median")
    # llmprobe leaves the top-level block null on a noisy predictable/novel pair:
    # the shortest context rung carries the same measurement.
    rungs = bench.get("contextScaling") or [{}]
    tps = ((bench.get("speculative") or {}).get("tokensPerStep")
           or (rungs[0].get("speculative") or {}).get("tokensPerStep") or 1.0)
    # WHICH speculative mode ran is only knowable from the server's own log
    # (llmprobe reports that one engaged, not which one). Name it in the cell
    # only when it actually paid: armed-but-not-accepting is not "mtp".
    log = path.with_suffix(".log")
    modes = re.findall(r"\[spec-stats\] mode=(\w+)",
                       log.read_text(errors="replace")) if log.exists() else []
    mode = f" {max(set(modes), key=modes.count)}" if modes and tps > 1.05 else ""
    if decode is None:
        print(f"| {path.stem} | · |  (no bench block)")
        continue
    print(f"| {path.stem} | {decode:.0f}{mode} |"
          f"  (prefill {prefill:.0f}, {tps:.2f} tok/step)")
PY
echo
echo "=== reports $OUT"
