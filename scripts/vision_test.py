"""Verify the vision tower survived quantization.

The 333 visual.* tensors were deliberately left BF16, but "present in the file" is not
"still works" -- the merger and patch_embed have to line up with the quantized language
model, and the processor config had to survive the save. So this renders images whose
content is unambiguous and checks the model reads them back exactly.

Three probes, increasingly demanding:
  1. a large printed number   -> OCR-ish, fails loudly if the tower is broken
  2. coloured shapes          -> colour + shape + counting
  3. a tiny bar chart         -> relative magnitude, needs the merger to be sane
"""
import base64, io, json, os, sys, urllib.request

URL = os.environ.get("VLLM_URL", "http://127.0.0.1:8899") + "/v1/chat/completions"
MODEL = os.environ.get("MODEL", "qwen3.8-27b-uncensored-jc")

from PIL import Image, ImageDraw, ImageFont


def font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def number_img(text="738261"):
    im = Image.new("RGB", (640, 260), "white")
    d = ImageDraw.Draw(im)
    d.text((40, 70), text, fill="black", font=font(120))
    return im


def shapes_img():
    im = Image.new("RGB", (640, 400), "white")
    d = ImageDraw.Draw(im)
    d.ellipse([40, 40, 200, 200], fill="red")
    d.ellipse([240, 40, 400, 200], fill="red")
    d.ellipse([440, 40, 600, 200], fill="red")
    d.rectangle([40, 240, 200, 360], fill="blue")
    d.rectangle([240, 240, 400, 360], fill="blue")
    return im   # 3 red circles, 2 blue squares


def chart_img():
    im = Image.new("RGB", (640, 400), "white")
    d = ImageDraw.Draw(im)
    vals = [("A", 60), ("B", 180), ("C", 110), ("D", 300)]
    x = 60
    for label, h in vals:
        d.rectangle([x, 360 - h, x + 90, 360], fill="steelblue")
        d.text((x + 30, 365), label, fill="black", font=font(28))
        x += 140
    d.line([40, 360, 620, 360], fill="black", width=3)
    return im   # D tallest, then B, then C, then A


def b64(im):
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def ask(img, question, label, expect):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": "data:image/png;base64," + b64(img)}},
            {"type": "text", "text": question},
        ]}],
        "max_tokens": 512,
        "temperature": 0,
    }
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    try:
        d = json.load(urllib.request.urlopen(req, timeout=600))
    except Exception as e:
        body_txt = e.read()[:300].decode(errors="replace") if hasattr(e, "read") else str(e)
        print(f"  {label:14s} ERROR {type(e).__name__}: {body_txt}")
        return False
    m = d["choices"][0]["message"]
    ans = (m.get("content") or "").strip()
    ok = all(e.lower() in ans.lower() for e in expect)
    print(f"  {label:14s} {'PASS' if ok else 'FAIL'}  expect={expect}")
    print(f"                 answer: {ans[:220]!r}")
    return ok


print(f"vision probes against {URL}\n")
r = []
r.append(ask(number_img(), "What number is written in this image? Reply with the digits only.",
             "printed number", ["738261"]))
r.append(ask(shapes_img(), "How many red circles and how many blue squares are in this image? "
                           "Answer like: '<n> red circles, <m> blue squares'.",
             "shapes+count", ["3", "2"]))
r.append(ask(chart_img(), "In this bar chart, which bar is tallest and which is shortest? "
                          "Answer with the two letters.",
             "bar chart", ["D", "A"]))
print(f"\nVISION: {sum(r)}/{len(r)} passed")
print("VISION_OK" if all(r) else "VISION_DEGRADED")
