#!/usr/bin/env python3
"""Score a SERVED Qwen3.8-Flash-Next pack against the bf16 reference logits written by
`qwen38_flash_next_imatrix_collect.py --reference` (held-out windows the imatrix never saw).

Per position the prefix is decoded to text, re-encoded to confirm it round-trips to the
same ids (else skipped), and sent to `/v1/completions` greedy for ONE token with
`logprobs: 20`; `usage.prompt_tokens` must equal the prefix length. Reported per slice:
  top1     the pack's greedy token == the reference argmax
  ref_p    mean reference probability of the pack's pick (1.0 = perfect; the reference
           argmax's own mean probability is printed as the ceiling)
No mlx-lm, no teacher forcing: the server has no prompt-logprob surface, so each
position is its own one-token request. Boot the server with `--prefix-cache-entries 0`:
a hybrid restore is not bit-identical (~0.2 nats on the top token here), and both
packs must be scored cold.

  python3 tests/qwen38_flash_next_score.py --ref ~/claude-tmp/qwen38-flash-next-23/ref.npz \
      --url http://127.0.0.1:11400 --model <id> [--limit N]
"""

import argparse
import json
import sys
import urllib.request
from collections import defaultdict

import numpy as np


def softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:11400")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True, help="dir with tokenizer.json (the pack)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(f"{args.tokenizer}/tokenizer.json")
    ref = np.load(args.ref, allow_pickle=True)
    n = len(ref["pos"]) if args.limit is None else min(args.limit, len(ref["pos"]))
    per = defaultdict(lambda: {"top1": [], "ref_p": [], "ceil": [], "kld": []})
    per_row = []  # one entry per scored reference position, so two packs compare on the rows both answered
    skipped = {"roundtrip": 0, "prompt_tokens": 0, "no_logprobs": 0}
    for i in range(n):
        ids = ref["ids"][i].astype(np.int64).tolist()
        text = tok.decode(ids, skip_special_tokens=False)
        if tok.encode(text, add_special_tokens=False).ids != ids:
            skipped["roundtrip"] += 1
            continue
        body = json.dumps({"model": args.model, "prompt": text, "max_tokens": 1, "temperature": 0,
                           "logprobs": 1024, "prompt_cache_key": "score"}).encode()
        req = urllib.request.Request(f"{args.url}/v1/completions", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            resp = json.load(r)
        if resp["usage"]["prompt_tokens"] != len(ids):
            skipped["prompt_tokens"] += 1
            continue
        lp = resp["choices"][0].get("logprobs")
        if not lp or not lp.get("tokens"):
            skipped["no_logprobs"] += 1
            continue
        got = lp["tokens"][0]
        logits = ref["logits"][i].astype(np.float32)
        p = softmax(logits)
        ref_arg = int(logits.argmax())
        ref_str = tok.decode([ref_arg], skip_special_tokens=False)
        # the pack's pick as a reference probability: match its string among the reference top-20
        top20 = np.argpartition(-p, 20)[:20]
        pick_p = 0.0
        for t in top20:
            if tok.decode([int(t)], skip_special_tokens=False) == got:
                pick_p = float(p[t])
                break
        # KL(bf16 || pack) over the reference's top-1024 support: the pack's logprob per
        # token comes from its returned top list, tokens it left out take its last entry.
        q = lp["top_logprobs"][0] if lp.get("top_logprobs") else {}
        floor = min(q.values()) if q else -30.0
        top_m = np.argpartition(-p, 1024)[:1024]
        top_m = top_m[np.argsort(-p[top_m])]
        kl = kl20 = 0.0
        for n, t in enumerate(top_m):
            qt = q.get(tok.decode([int(t)], skip_special_tokens=False), floor)
            term = float(p[t]) * (float(np.log(max(p[t], 1e-30))) - qt)
            kl += term
            if n < 20:  # the support an engine capped at 20 logprobs can be compared on
                kl20 += term
        s = str(ref["slice"][i])
        per_row.append({"i": i, "slice": s, "top1": float(got == ref_str), "ref_p": pick_p, "kld": kl, "kld20": kl20})
        per[s]["top1"].append(float(got == ref_str))
        per[s]["ref_p"].append(pick_p)
        per[s]["ceil"].append(float(p[ref_arg]))
        per[s]["kld"].append(kl)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{n}", flush=True)
    rows = {k: {"top1": float(np.mean(v["top1"])), "ref_p": float(np.mean(v["ref_p"])),
                "ref_p_ceiling": float(np.mean(v["ceil"])), "kld": float(np.mean(v["kld"])), "n": len(v["top1"])} for k, v in sorted(per.items())}
    allv = [x for v in per.values() for x in v["top1"]]
    allp = [x for v in per.values() for x in v["ref_p"]]
    allc = [x for v in per.values() for x in v["ceil"]]
    allk = [x for v in per.values() for x in v["kld"]]
    rows["all"] = {"top1": float(np.mean(allv)), "ref_p": float(np.mean(allp)), "ref_p_ceiling": float(np.mean(allc)), "kld": float(np.mean(allk)), "n": len(allv)}
    print(f"model {args.model}: skipped {skipped}")
    for k, r in rows.items():
        print(f"  {k:6s} n={r['n']:4d} top1 {r['top1']*100:5.1f}%  ref_p {r['ref_p']:.3f} (ceiling {r['ref_p_ceiling']:.3f})  kld {r['kld']:.4f}")
    if args.out:
        json.dump({"model": args.model, "skipped": skipped, "rows": rows, "per_row": per_row}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
