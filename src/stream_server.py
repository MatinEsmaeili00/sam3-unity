#!/usr/bin/env python3
"""
SAM 3 - Live Stream Segmentation Server (WebSocket)
=====================================================

Clients on the local network (e.g. a Unity app on a Meta Quest 3) stream
camera frames as JPEG and receive, per frame, the SAM 3 masks for a set of
text prompts. SAM 3's video predictor needs the whole video up front, so a
live stream is served with the image model, one frame at a time.

Protocol (endpoint: ws://<host>:<port>/ws[?token=...])
------------------------------------------------------
client -> server
  text   {"type": "config", "prompts": ["cup", "keyboard"], "threshold": 0.5}
  binary 4-byte little-endian uint32 frame_id, followed by JPEG bytes
server -> client
  text   {"type": "ready" | "config_ack" | "error", ...}
  text   {"type": "result", "frame_id", "width", "height", "inference_ms",
          "detections": [{"prompt", "prompt_index", "score",
                          "box": [x0, y0, x1, y1] (normalized, origin top-left),
                          "color": [r, g, b]}],
          "mask_png": base64 RGBA PNG overlay (same size as the frame) or null}

If frames arrive faster than the GPU can process them, only the newest one
is kept, so latency never builds up. Clients get the best latency by sending
the next frame only after the previous result has arrived.

Usage
-----
    python src/stream_server.py --host 0.0.0.0 --port 8765 [--token SECRET]
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import struct
import time
from concurrent.futures import ThreadPoolExecutor

import torch
import uvicorn
from fastapi import FastAPI, WebSocket
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

PALETTE = [
    (230, 25, 75), (60, 180, 75), (0, 130, 200), (255, 225, 25), (145, 30, 180),
    (70, 240, 240), (245, 130, 48), (240, 50, 230), (210, 245, 60), (250, 190, 212),
]
MASK_ALPHA = 150
MAX_PROMPTS = 10
MAX_DETECTIONS = 20

# All GPU work runs on this single thread: it serializes access to the one
# model and keeps torch.autocast (which is thread-local) in one place.
GPU = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sam3-gpu")
processor: Sam3Processor | None = None
auth_token: str | None = None
app = FastAPI(title="SAM 3 stream server")


def load_processor() -> Sam3Processor:
    return Sam3Processor(build_sam3_image_model())


@torch.inference_mode()
def run_frame(proc: Sam3Processor, jpeg: bytes, prompts: list[str], threshold: float) -> dict:
    image = Image.open(io.BytesIO(jpeg)).convert("RGB")
    w, h = image.size
    t0 = time.perf_counter()
    detections = []
    mask_png = None

    if prompts:
        proc.confidence_threshold = threshold
        overlay = torch.zeros((h, w, 4), dtype=torch.uint8, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state = proc.set_image(image)
            for pi, prompt in enumerate(prompts):
                out = proc.set_text_prompt(prompt=prompt, state=state)
                color = PALETTE[pi % len(PALETTE)]
                rgba = torch.tensor([*color, MASK_ALPHA], dtype=torch.uint8, device="cuda")
                scores = out["scores"].float()
                for i in torch.argsort(scores, descending=True).tolist():
                    if len(detections) >= MAX_DETECTIONS:
                        break
                    overlay[out["masks"][i, 0]] = rgba
                    x0, y0, x1, y1 = out["boxes"][i].float().tolist()
                    detections.append({
                        "prompt": prompt,
                        "prompt_index": pi,
                        "score": round(float(scores[i]), 4),
                        "box": [x0 / w, y0 / h, x1 / w, y1 / h],
                        "color": list(color),
                    })
        buf = io.BytesIO()
        Image.fromarray(overlay.cpu().numpy(), "RGBA").save(buf, format="PNG", compress_level=1)
        mask_png = base64.b64encode(buf.getvalue()).decode("ascii")

    return {
        "width": w,
        "height": h,
        "inference_ms": round((time.perf_counter() - t0) * 1000, 1),
        "detections": detections,
        "mask_png": mask_png,
    }


def parse_config(cmd: dict, cfg: dict) -> None:
    prompts = cmd.get("prompts", cfg["prompts"])
    if not isinstance(prompts, list) or not all(isinstance(p, str) for p in prompts):
        raise ValueError("'prompts' must be a list of strings")
    cfg["prompts"] = [p.strip() for p in prompts if p.strip()][:MAX_PROMPTS]
    cfg["threshold"] = min(max(float(cmd.get("threshold", cfg["threshold"])), 0.0), 1.0)


@app.get("/")
def health():
    return {
        "status": "ok",
        "model": "facebook/sam3 (image)",
        "gpu": torch.cuda.get_device_name(0),
        "websocket": "/ws",
        "auth": auth_token is not None,
    }


@app.websocket("/ws")
async def stream(ws: WebSocket):
    if auth_token and ws.query_params.get("token") != auth_token:
        await ws.close(code=1008)
        return
    await ws.accept()
    client = f"{ws.client.host}:{ws.client.port}" if ws.client else "?"
    print(f"[connect] {client}", flush=True)

    cfg = {"prompts": [], "threshold": 0.5}
    latest: dict = {"frame": None}
    new_frame = asyncio.Event()

    async def infer_loop():
        loop = asyncio.get_running_loop()
        try:
            while True:
                await new_frame.wait()
                new_frame.clear()
                frame_id, jpeg = latest["frame"]
                try:
                    result = await loop.run_in_executor(
                        GPU, run_frame, processor, jpeg, list(cfg["prompts"]), cfg["threshold"]
                    )
                except Exception as e:  # bad JPEG, CUDA OOM, ...: report and keep serving
                    await ws.send_json({"type": "error", "frame_id": frame_id, "message": str(e)})
                    continue
                await ws.send_json({"type": "result", "frame_id": frame_id, **result})
        except Exception:
            pass  # socket closed while sending; the receive loop handles cleanup

    task = asyncio.create_task(infer_loop())
    await ws.send_json({"type": "ready", "max_prompts": MAX_PROMPTS})
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            data = msg.get("bytes")
            if data is not None:
                if len(data) > 4:
                    latest["frame"] = (struct.unpack_from("<I", data)[0], data[4:])
                    new_frame.set()
            elif msg.get("text"):
                try:
                    cmd = json.loads(msg["text"])
                    if cmd.get("type") != "config":
                        raise ValueError(f"unknown message type: {cmd.get('type')!r}")
                    parse_config(cmd, cfg)
                    print(f"[config] {client} prompts={cfg['prompts']} threshold={cfg['threshold']}", flush=True)
                    await ws.send_json({"type": "config_ack", **cfg})
                except (ValueError, TypeError) as e:
                    await ws.send_json({"type": "error", "message": str(e)})
    finally:
        task.cancel()
        print(f"[disconnect] {client}", flush=True)


def main(argv=None):
    global processor, auth_token
    parser = argparse.ArgumentParser(description="SAM 3 live stream segmentation server")
    parser.add_argument("--host", default="0.0.0.0", help="Interface to bind (0.0.0.0 = all, reachable from the LAN)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--token", default=os.environ.get("SAM3_TOKEN") or None,
        help="Optional shared secret (default: $SAM3_TOKEN); clients must connect with ?token=...",
    )
    args = parser.parse_args(argv)
    auth_token = args.token

    print("Loading SAM 3 (downloads the checkpoint on first run)...", flush=True)
    processor = GPU.submit(load_processor).result()
    warmup = io.BytesIO()
    Image.new("RGB", (640, 480), (128, 128, 128)).save(warmup, format="JPEG")
    GPU.submit(run_frame, processor, warmup.getvalue(), ["object"], 0.5).result()
    print(f"Model ready. Listening on ws://{args.host}:{args.port}/ws", flush=True)

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
