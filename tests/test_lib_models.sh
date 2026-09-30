#!/bin/bash
# Hermetic tests for tests/_lib_models.sh: model lookup across roots and the
# GPU-budget gate the live scripts skip on. Scratch dirs only, no weights.
#
# Run: bash tests/test_lib_models.sh
set -u
source "$(dirname "$0")/_lib_models.sh"

PASS=0; FAIL=0
ok() { # name  actual  expected
    if [ "$2" = "$3" ]; then PASS=$((PASS + 1)); echo "  PASS $1"
    else FAIL=$((FAIL + 1)); echo "  FAIL $1 — expected [$3], got [$2]"; fi
}

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
# Never the tree's real server: a pack past the budget asks MLX_SERVE_BIN for its load bill (model_load_gb).
export MLX_SERVE_BIN="$TMP/no-binary"
mkdir -p "$TMP/a/org/small" "$TMP/b/org/small" "$TMP/b/org/big" "$TMP/b/gguf-dir"
head -c 2048 /dev/zero > "$TMP/a/org/small/model.safetensors"
touch "$TMP/b/gguf-dir/m.gguf"
mkdir -p "$TMP/b/org/linked"
ln -s "$TMP/a/org/small/model.safetensors" "$TMP/b/org/linked/model.safetensors"
export MLX_SERVE_MODEL_ROOTS="$TMP/a:$TMP/b"

echo "── find_model ──"
ok "first root wins for one candidate"   "$(find_model org/small)"               "$TMP/a/org/small"
ok "candidate order beats root order"    "$(find_model org/big org/small)"       "$TMP/b/org/big"
ok "falls through to a later candidate"  "$(find_model org/none org/small)"      "$TMP/a/org/small"
ok "a file path resolves (gguf)"         "$(find_model gguf-dir/m.gguf)"         "$TMP/b/gguf-dir/m.gguf"
ok "miss prints nothing"                 "$(find_model org/none)"                ""
find_model org/none >/dev/null; ok "miss returns 1" "$?" "1"

echo "── model_roots ──"
ok "env roots replace the defaults"      "$(model_roots | tr '\n' ' ')"          "$TMP/a $TMP/b "

echo "── sizes and budget ──"
ok "a partial GB rounds up to 1"         "$(model_gb "$TMP/a/org/small")"        "1"
ok "an empty dir is 0 GB"                "$(model_gb "$TMP/b/org/big")"          "0"
ok "size follows a symlinked shard"      "$(model_gb "$TMP/b/org/linked")"       "1"
MAX_MODEL_GB=1 model_fits "$TMP/a/org/small"; ok "fits at the cap"   "$?" "0"
MAX_MODEL_GB=0 model_fits "$TMP/a/org/small"; ok "past the cap"      "$?" "1"
ok "MAX_MODEL_GB overrides the budget"   "$(MAX_MODEL_GB=7 max_model_gb)"        "7"
b="$(gpu_budget_gb)"
ok "GPU budget is a positive integer"    "$([[ "$b" =~ ^[0-9]+$ && "$b" -gt 0 ]] && echo yes)" "yes"
# The cap mirrors the server's load preflight: weights + min(weights/8, 6) + 1 GB.
gpu_budget_gb() { echo 14; }
ok "14 GB budget: an 11 GB pack is the largest" "$(max_model_gb)"                "11"
ok "96 GB budget: headroom caps at 7 GB"         "$(gpu_budget_gb() { echo 96; }; max_model_gb)" "89"
ok "extra headroom for a big prompt's KV"        "$(MODEL_HEADROOM_GB=6 max_model_gb)" "6"

echo "── model_load_gb (a streamed pack's own load bill) ──"
# Stub servers: one answers with an arch bill, one bills shards, one fails; none loads anything.
printf '#!/bin/bash\necho "arch 0"\n' > "$TMP/bin-arch"; printf '#!/bin/bash\necho "shards 2048"\n' > "$TMP/bin-shards"
printf '#!/bin/bash\nexit 1\n' > "$TMP/bin-fail"; chmod +x "$TMP/bin-arch" "$TMP/bin-shards" "$TMP/bin-fail"
ok "a pack within budget is its shards (the binary is not asked)" "$(MAX_MODEL_GB=1 MLX_SERVE_BIN="$TMP/bin-fail" model_load_gb "$TMP/a/org/small")" "1"
ok "past the budget: the arch's own bill"      "$(MAX_MODEL_GB=0 MLX_SERVE_BIN="$TMP/bin-arch" model_load_gb "$TMP/a/org/small")" "0"
ok "past the budget, a shard-billed arch"      "$(MAX_MODEL_GB=0 MLX_SERVE_BIN="$TMP/bin-shards" model_load_gb "$TMP/a/org/small")" "1"
ok "past the budget, the binary fails: shards" "$(MAX_MODEL_GB=0 MLX_SERVE_BIN="$TMP/bin-fail" model_load_gb "$TMP/a/org/small")" "1"
ok "past the budget, no binary: shards"        "$(MAX_MODEL_GB=0 MLX_SERVE_BIN="$TMP/none" model_load_gb "$TMP/a/org/small")" "1"
MAX_MODEL_GB=0 MLX_SERVE_BIN="$TMP/bin-arch" model_fits "$TMP/a/org/small"; ok "a streamed pack fits by its bill" "$?" "0"
MAX_MODEL_GB=0 MLX_SERVE_BIN="$TMP/bin-shards" model_fits "$TMP/a/org/small"; ok "a shard-billed pack past the budget" "$?" "1"

echo "── find_fitting_model ──"
head -c 2048 /dev/zero > "$TMP/b/org/big/model.safetensors"
ok "skips a present pack past the budget" "$(MAX_MODEL_GB=0 find_fitting_model org/small org/big)" ""
ok "first fitting candidate wins"         "$(MAX_MODEL_GB=1 find_fitting_model org/none org/big org/small)" "$TMP/b/org/big"
MAX_MODEL_GB=0 find_fitting_model org/small >/dev/null; ok "nothing fits returns 1" "$?" "1"

unset MLX_SERVE_MODEL_ROOTS
echo "── default roots ──"
ok "~/.mlx-serve/models listed first when present" \
    "$(HOME="$TMP/home" bash -c 'mkdir -p "$HOME/.mlx-serve/models"; source "'"$(dirname "$0")"'/_lib_models.sh"; model_roots | head -1')" \
    "$TMP/home/.mlx-serve/models"

echo ""
TOTAL=$((PASS + FAIL))
if [ "$FAIL" -eq 0 ]; then echo "PASS $TOTAL/$TOTAL"; exit 0
else echo "FAIL $FAIL/$TOTAL"; exit 1; fi
