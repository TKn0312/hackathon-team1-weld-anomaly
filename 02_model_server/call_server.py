"""Wake the model server and call its /predict API in the two ways another program can.

    python call_server.py path/to/tile.png                              # the Render server
    python call_server.py path/to/tile.png --url http://127.0.0.1:7860  # a local server

1. wake_up: a free Render service sleeps after 15 minutes; the first request wakes it (about a minute).
2. gradio_client: the easy way from Python.
3. plain HTTP, two steps: exactly what the Cloudflare app does in JavaScript.
Run it before a demo (Friday: before 10:00) so the server is awake.
"""
import argparse
import base64
import json
import time

import httpx
from gradio_client import Client

RENDER_URL = "https://hackathon-team1-weld-anomaly.onrender.com/gradio"   # the tile app and its /predict API


def wake_up(url, tries=12, wait=10):
    start = time.time()
    for _ in range(tries):
        try:
            if httpx.get(url, timeout=60).status_code == 200:
                print(f"server awake after {time.time() - start:.0f} s")
                return True
        except httpx.HTTPError:
            pass
        time.sleep(wait)
    print("server did not answer: check the Render dashboard (Deploys, Logs)")
    return False


def call_with_gradio_client(url, image_b64):
    return Client(url, verbose=False).predict(image_b64, api_name="/predict")


def call_with_http(url, image_b64, timeout=90):
    # Step 1: send the input, get a ticket (event_id)
    event_id = httpx.post(f"{url}/gradio_api/call/predict", json={"data": [image_b64]}, timeout=timeout).json()["event_id"]
    # Step 2: fetch the result; it comes after the line "event: complete"
    text = httpx.get(f"{url}/gradio_api/call/predict/{event_id}", timeout=timeout).text
    lines = text.splitlines()
    if "event: complete" not in lines:
        raise RuntimeError(f"the server did not complete the request:\n{text[:500]}")
    data_line = lines[lines.index("event: complete") + 1]
    return json.loads(data_line.removeprefix("data:").strip())[0]


def summary(result):
    return (f"risk {result['risk_level']:5s} | score {result['score']:.6f} | "
            f"{result['score_vs_threshold']}x the 95% threshold | anomaly {result['is_anomaly']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", help="an X-ray tile (PNG or JPEG)")
    parser.add_argument("--url", default=RENDER_URL, help=f"server address (default: {RENDER_URL})")
    args = parser.parse_args()

    image_b64 = base64.b64encode(open(args.image, "rb").read()).decode()
    if wake_up(args.url):
        t = time.time(); r1 = call_with_gradio_client(args.url, image_b64)
        print(f"gradio_client : {summary(r1)}  ({time.time() - t:.1f} s)")
        t = time.time(); r2 = call_with_http(args.url, image_b64)
        print(f"plain HTTP    : {summary(r2)}  ({time.time() - t:.1f} s)")
        print("same answer both ways:", r1["score"] == r2["score"] and r1["risk_level"] == r2["risk_level"])
