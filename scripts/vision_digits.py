"""Characterise the OCR miss: is the vision path broken, or is this a length limit?

The tower is BF16 and untouched, so a failure here would have to come from the FP4
language model reading its features. Sweeping digit-count separates "broken" (fails
everywhere) from "precision limit on dense strings" (degrades with length).
Each length is asked twice with different digits to avoid reading luck as signal.
"""
import base64, io, json, os, urllib.request
from PIL import Image, ImageDraw, ImageFont

URL = os.environ.get("VLLM_URL", "http://127.0.0.1:8899") + "/v1/chat/completions"
MODEL = os.environ.get("MODEL", "qwen3.8-27b-uncensored-jc")

CASES = [("3-digit", "482"), ("3-digit", "917"),
         ("4-digit", "5093"), ("4-digit", "2748"),
         ("6-digit", "738261"), ("6-digit", "405913"),
         ("8-digit", "60274183"), ("8-digit", "91538402")]


def font(sz):
    p = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    return ImageFont.truetype(p, sz) if os.path.exists(p) else ImageFont.load_default()


def img(text):
    w = 180 + 110 * len(text)
    im = Image.new("RGB", (w, 260), "white")
    ImageDraw.Draw(im).text((60, 70), text, fill="black", font=font(120))
    buf = io.BytesIO(); im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


ok = 0
for label, digits in CASES:
    body = {"model": MODEL, "max_tokens": 256, "temperature": 0,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + img(digits)}},
                {"type": "text", "text": "What number is in this image? Reply with the digits only."}]}]}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        d = json.load(urllib.request.urlopen(req, timeout=600))
        ans = (d["choices"][0]["message"].get("content") or "").strip()
    except Exception as e:
        ans = "ERROR %s" % type(e).__name__
    got = "".join(c for c in ans if c.isdigit())
    good = got == digits
    ok += good
    print("  %-8s expect %-9s got %-11s %s" % (label, digits, got or repr(ans)[:11],
                                               "OK" if good else "MISS"))
print("\n  %d/%d exact" % (ok, len(CASES)))
