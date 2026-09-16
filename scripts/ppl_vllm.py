"""Perplexity through the live vLLM serving path, via prompt_logprobs.

Measuring in transformers would test a different code path than the one we serve on.
Using vLLM's prompt_logprobs puts the actual FP4/FP8 kernels in the loop, which is the
number that matters: "what does this checkpoint do when served".

Same corpus as the upstream author's own comparison (wikitext-2-raw), fixed chunk length,
identical chunks for every model so the rows are paired rather than independent samples.

  MODEL=... VLLM_URL=http://127.0.0.1:8899 python3 ppl_vllm.py <label>
"""
import json, math, os, sys, urllib.request

URL = os.environ.get("VLLM_URL", "http://127.0.0.1:8899")
MODEL = os.environ.get("MODEL", "m")
LABEL = sys.argv[1] if len(sys.argv) > 1 else "run"
NCHUNK = int(os.environ.get("NCHUNK", "24"))
CHUNKTOK = int(os.environ.get("CHUNKTOK", "2048"))
CACHE = os.path.expanduser("~/ppl-chunks-%d-%d.json" % (NCHUNK, CHUNKTOK))


def tokenize(text):
    req = urllib.request.Request(
        URL + "/tokenize", data=json.dumps({"model": MODEL, "prompt": text}).encode(),
        headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=300))["tokens"]


def detokenize(ids):
    req = urllib.request.Request(
        URL + "/detokenize", data=json.dumps({"model": MODEL, "tokens": ids}).encode(),
        headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=300))["prompt"]


# Build chunks once and reuse them for every model, so comparisons are paired.
if os.path.exists(CACHE):
    chunks = json.load(open(CACHE))
    print("reusing %d cached chunks" % len(chunks))
else:
    from datasets import load_dataset
    raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n".join(raw["text"])
    ids = tokenize(text)
    chunks = []
    for i in range(NCHUNK):
        seg = ids[i * CHUNKTOK:(i + 1) * CHUNKTOK]
        if len(seg) < CHUNKTOK:
            break
        chunks.append(detokenize(seg))
    json.dump(chunks, open(CACHE, "w"))
    print("built %d chunks of %d tokens" % (len(chunks), CHUNKTOK))

tot_lp, tot_n = 0.0, 0
for i, c in enumerate(chunks):
    body = {"model": MODEL, "prompt": c, "max_tokens": 1,
            "temperature": 0, "prompt_logprobs": 0}
    req = urllib.request.Request(
        URL + "/v1/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=1800))
    pl = d["choices"][0].get("prompt_logprobs") or []
    n = 0
    s = 0.0
    for entry in pl:
        if not entry:
            continue          # first token has no conditional logprob
        # entry maps token_id -> {logprob, ...}; with prompt_logprobs=0 it holds the
        # actual token only
        for _tid, info in entry.items():
            s += info["logprob"]
            n += 1
            break
    tot_lp += s
    tot_n += n
    if (i + 1) % 6 == 0:
        print("  %2d/%d chunks  running ppl=%.4f" % (i + 1, len(chunks),
                                                     math.exp(-tot_lp / max(tot_n, 1))), flush=True)

ppl = math.exp(-tot_lp / max(tot_n, 1))
print("\n%-28s tokens=%d  mean_logprob=%.6f  PPL=%.4f" % (LABEL, tot_n, tot_lp / tot_n, ppl))
print("PPL_RESULT %s %.4f %d" % (LABEL, ppl, tot_n))
