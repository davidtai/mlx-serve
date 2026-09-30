#!/bin/bash
# Shared model lookup for the live test scripts, so one tree runs on any Mac.
# A script names a checkpoint by its <org>/<repo> (or <dir>/<file>.gguf) path
# RELATIVE to a model root; the lib finds it on whatever roots this box has and
# says whether it fits this box's GPU budget.
#
#   source "$(dirname "$0")/_lib_models.sh"
#   MODEL="${MY_TEST_MODEL:-$(find_model mlx-community/gemma-4-e4b-it-8bit mlx-community/gemma-4-e4b-it-4bit)}"
#   model_fits "$MODEL" || { echo "SKIP: $(model_gb "$MODEL") GB > $(max_model_gb) GB"; exit 0; }
#
# Roots: MLX_SERVE_MODEL_ROOTS (colon-separated) replaces the defaults, which are
# ~/.mlx-serve/models, ~/.lmstudio/models and models|models-dl|Models|gguf on
# every mounted volume. Budget: MAX_MODEL_GB overrides the derived one. A pack that streams its experts is sized by
# the load bill the server admits it on (model_load_gb), not its shards.

model_roots() {
    if [[ -n "${MLX_SERVE_MODEL_ROOTS:-}" ]]; then
        tr ':' '\n' <<< "$MLX_SERVE_MODEL_ROOTS"
        return
    fi
    local d s seen=()
    for d in "$HOME/.mlx-serve/models" "$HOME/.lmstudio/models" /Volumes/*/models /Volumes/*/models-dl /Volumes/*/Models /Volumes/*/gguf; do
        [[ -d "$d" ]] || continue
        # a case-insensitive volume answers to both models/ and Models/
        for s in ${seen[@]+"${seen[@]}"}; do [[ "$d" -ef "$s" ]] && continue 2; done
        seen+=("$d"); echo "$d"
    done
}

# First existing <root>/<rel> over every candidate, in the order given;
# prints nothing and returns 1 when none is on this box.
find_model() {
    local rel root
    for rel in "$@"; do
        while IFS= read -r root; do
            [[ -e "$root/$rel" ]] && { echo "$root/$rel"; return 0; }
        done < <(model_roots)
    done
    return 1
}

# On-disk size in whole GB, rounded up, following symlinked shards.
model_gb() {
    local kb
    kb=$(du -skL "$1" 2>/dev/null | cut -f1)
    echo $(( (${kb:-0} + 1048575) / 1048576 ))
}

# Metal's working set: iogpu.wired_limit_mb when it is raised, else the macOS
# default of about 3/4 of RAM.
gpu_budget_gb() {
    local mb
    mb=$(sysctl -n iogpu.wired_limit_mb 2>/dev/null)
    [[ "${mb:-0}" -gt 0 ]] || mb=$(( $(sysctl -n hw.memsize) / 1048576 * 3 / 4 ))
    echo $(( mb / 1024 ))
}

# Largest pack the server's load preflight admits on this box: it wants
# weights + min(weights/8, 6) + 1 GB (scheduler.loadRequirementBytes).
# MODEL_HEADROOM_GB adds room on top, for a test whose prompt needs big KV.
max_model_gb() {
    [[ -n "${MAX_MODEL_GB:-}" ]] && { echo "$MAX_MODEL_GB"; return; }
    local room=$(( $(gpu_budget_gb) - 1 - ${MODEL_HEADROOM_GB:-0} ))
    if (( room >= 54 )); then echo $(( room - 6 )); else echo $(( room * 8 / 9 )); fi
}

# The GB the server's load pre-flight bills a pack: its shards, unless they are past the budget and the arch bills its
# own load (streamed experts: the resident set, not the expert bank), which the server binary states without loading
# anything (`mlx-serve --model <dir> --print-load-bytes` -> "arch <bytes>"). MLX_SERVE_BIN, else BINARY, else the
# tree's build names the binary; without one the shards stand.
model_load_gb() {
    local gb bin out
    gb=$(model_gb "$1")
    if [[ "$gb" -gt "$(max_model_gb)" ]]; then
        bin="${MLX_SERVE_BIN:-${BINARY:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/zig-out/bin/mlx-serve}}"
        if [[ -x "$bin" ]] && out=$("$bin" --model "$1" --print-load-bytes 2>/dev/null | tail -1) && [[ "$out" =~ ^arch\ ([0-9]+)$ ]]; then
            gb=$(( (${BASH_REMATCH[1]} + 1073741823) / 1073741824 ))
        fi
    fi
    echo "$gb"
}

model_fits() {
    [[ "$(model_load_gb "$1")" -le "$(max_model_gb)" ]]
}

# Like find_model, but a candidate past the budget is passed over for the next
# one. A test with a big prompt raises MODEL_HEADROOM_GB for its KV.
find_fitting_model() {
    local rel m
    for rel in "$@"; do
        m=$(find_model "$rel") && model_fits "$m" && { echo "$m"; return 0; }
    done
    return 1
}
