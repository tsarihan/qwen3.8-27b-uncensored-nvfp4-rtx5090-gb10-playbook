"""TTFT / prefill probe at increasing prompt lengths.

This is the test a concurrency ladder CANNOT do. sweep.py sends ~133-token prompts, which
sit inside a single prefill chunk -- exactly the regime where vLLM 0.26 MTP looked great
(+59% decode) while collapsing >20x on anything longer. Finding A measured k=1..4 at
ctx 16471 going to TTFT 68-73s / ~240 tok/s prefill against k=0's 3.1s / 5356 tok/s.

So: prompts deliberately straddle the --max-num-batched-tokens=8192 chunk boundary.
4K stays inside one chunk; 16K and 64K do not. A healthy config keeps prefill roughly
flat across all three. A collapsing one falls off a cliff after the first rung.
"""
import json, os, sys, time, urllib.request

URL = os.environ.get("VLLM_URL", "http://127.0.0.1:8138") + "/v1/chat/completions"
TOK = os.environ.get("VLLM_URL", "http://127.0.0.1:8138") + "/tokenize"
MODEL = os.environ.get("MODEL", "qwen3.8:27b-orca")
LABEL = os.environ.get("LABEL", "run")

UNIT = ("The maintenance log records routine calibration of the sensor array. "
        "Ambient conditions remained nominal throughout the observation window. ")


def ntok(text):
    req = urllib.request.Request(
        TOK, data=json.dumps({"model": MODEL, "prompt": text}).encode(),
        headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=180))["count"]


def filler(target):
    s = UNIT * max(1, target // 12)
    for _ in range(8):
        n = ntok(s)
        if abs(n - target) <= target * 0.02:
            break
        s = s[: max(1, int(len(s) * target / max(n, 1)))]
    return s, ntok(s)


print("%-6s %10s %9s %12s %10s" % ("target", "prompt_tok", "ttft_s", "prefill_t/s", "decode_t/s"))
for target in (4096, 16384, 65536):
    text, actual = filler(target)
    body = {"model": MODEL,
            "messages": [{"role": "user", "content": text + "\n\nReply with exactly: ok"}],
            "max_tokens": 64, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = None
    usage = None
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                d = json.loads(payload)
                if d.get("usage"):
                    usage = d["usage"]
                ch = d.get("choices") or []
                if ch:
                    delta = ch[0].get("delta", {})
                    if first is None and (delta.get("content") or delta.get("reasoning")
                                          or delta.get("reasoning_content")):
                        first = time.perf_counter()
        done = time.perf_counter()
    except Exception as e:
        print("%-6d %10d  ERROR %s: %s" % (target, actual, type(e).__name__, str(e)[:80]))
        continue
    if first is None:
        print("%-6d %10d  no first token" % (target, actual))
        continue
    ttft = first - t0
    out = (usage or {}).get("completion_tokens", 0)
    dec = (out - 1) / max(done - first, 1e-9) if out > 1 else 0.0
    print("%-6d %10d %9.2f %12.1f %10.2f" % (target, actual, ttft, actual / ttft, dec))
print("PROBE_DONE " + LABEL)
