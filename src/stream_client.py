#!/usr/bin/env python3
"""
Test client for stream_server.py - run it on any machine on the LAN to check
connectivity and measure end-to-end latency before wiring up Unity.

    pip install websockets pillow numpy opencv-python   # opencv only for --webcam
    python src/stream_client.py --url ws://192.168.1.50:8765/ws --prompts "person,cup"
    python src/stream_client.py --url ws://192.168.1.50:8765/ws --webcam 0 --frames 100
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import struct
import time

import numpy as np
from PIL import Image
from websockets.sync.client import connect


def jpeg_bytes(image: Image.Image, quality: int) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def main(argv=None):
    parser = argparse.ArgumentParser(description="SAM 3 stream server test client")
    parser.add_argument("--url", default="ws://localhost:8765/ws")
    parser.add_argument("--prompts", default="person,helmet,flag")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--image", default=None, help="Image to send repeatedly (default: bundled astronaut sample)")
    parser.add_argument("--webcam", type=int, default=None, help="Stream from this webcam index instead of an image")
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--width", type=int, default=640, help="Resize frames to this width before sending")
    parser.add_argument("--quality", type=int, default=80)
    parser.add_argument("--output", default="output/stream_client_last.png")
    args = parser.parse_args(argv)

    cap = None
    if args.webcam is not None:
        import cv2
        cap = cv2.VideoCapture(args.webcam)
        still = None
    elif args.image:
        still = Image.open(args.image).convert("RGB")
    else:
        from skimage import data
        still = Image.fromarray(data.astronaut())

    def next_frame() -> Image.Image:
        if cap is None:
            img = still
        else:
            ok, bgr = cap.read()
            if not ok:
                raise RuntimeError("webcam read failed")
            img = Image.fromarray(bgr[..., ::-1])
        if img.width != args.width:
            img = img.resize((args.width, round(img.height * args.width / img.width)))
        return img

    with connect(args.url, max_size=None) as ws:
        print("server:", json.loads(ws.recv()))
        prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
        ws.send(json.dumps({"type": "config", "prompts": prompts, "threshold": args.threshold}))
        print("server:", json.loads(ws.recv()))

        latencies, result, frame = [], None, None
        t_start = time.perf_counter()
        for frame_id in range(args.frames):
            frame = next_frame()
            t0 = time.perf_counter()
            ws.send(struct.pack("<I", frame_id) + jpeg_bytes(frame, args.quality))
            result = json.loads(ws.recv())
            if result["type"] != "result":
                print("server:", result)
                continue
            rtt = (time.perf_counter() - t0) * 1000
            latencies.append(rtt)
            found = ", ".join(f'{d["prompt"]}={d["score"]:.2f}' for d in result["detections"]) or "nothing"
            print(f"frame {frame_id}: round-trip {rtt:.0f} ms (gpu {result['inference_ms']:.0f} ms) -> {found}")
        elapsed = time.perf_counter() - t_start

    if latencies:
        print(f"\n{len(latencies)} frames, median round-trip {np.median(latencies):.0f} ms, "
              f"{len(latencies) / elapsed:.1f} FPS")
    if result and result.get("mask_png"):
        overlay = Image.open(io.BytesIO(base64.b64decode(result["mask_png"])))
        composite = frame.convert("RGBA")
        composite.alpha_composite(overlay)
        from pathlib import Path
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        composite.convert("RGB").save(args.output)
        print(f"[saved] {args.output}")
    if cap is not None:
        cap.release()


if __name__ == "__main__":
    main()
