# SAM 3 Unity - Live Segmentation Streaming for Meta Quest 3

Stream the **Meta Quest 3 passthrough camera** from a Unity VR app to
Meta's **Segment Anything Model 3 (SAM 3)** running on a GPU machine on
your network, and get text-prompted segmentation masks back live. Tell it
`"cup"` or `"keyboard"` and every matching object comes back as a colored
mask with a box and a confidence score.

```
Quest 3 (Unity)                                   GPU machine (Docker)
passthrough camera -> JPEG --- ws://IP:8765/ws ---> SAM 3 image model
RawImage overlay  <- mask PNG + boxes/scores <----- (one frame at a time)
```

This is the networked companion to
[sam3-claude](https://github.com/MatinEsmaeili00/sam3-claude), which covers
offline image segmentation and video tracking with SAM 3.

![stream result](examples/stream_result.png)

*A frame streamed over the network with prompts `person`, `helmet`, `flag`.
This is the overlay the Unity client receives, composited onto the frame.*

## ⚠️ Before you start

1. **A CUDA GPU** on the server machine. SAM 3 (848M parameters) is not
   practical on CPU.
2. **Your own Hugging Face access grant.** The weights are gated:
   - Request access at **https://huggingface.co/facebook/sam3**.
   - Create a **Read** token at https://huggingface.co/settings/tokens/new.
   - On the server machine, run once:
     ```bash
     pip install --user huggingface_hub
     hf auth login   # paste your token (input is hidden)
     ```
     Docker mounts `~/.cache/huggingface` into the container, so it inherits
     this login and caches the ~3.4GB checkpoint across runs.
3. **A Quest 3 or 3S on Horizon OS v74+**, which is required for
   passthrough camera access.

## 1. Start the server (GPU machine)

```bash
git clone https://github.com/MatinEsmaeili00/sam3-unity.git
cd sam3-unity
SAM3_TOKEN=pick-a-secret docker compose up --build
```

Or without Compose:

```bash
docker build -t sam3-unity:1.0.0 .
docker run --rm --gpus all -p 8765:8765 -e SAM3_TOKEN=pick-a-secret \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" sam3-unity:1.0.0
```

Wait for `Model ready. Listening on ws://0.0.0.0:8765/ws`, then get the
machine's IP with `hostname -I`. From another device (including the Quest's
browser), `http://<IP>:8765/` should return `{"status":"ok",...}`.

> **Use a token on any network you don't fully control.** Docker-published
> ports bypass `ufw`, so anyone who can reach the IP can otherwise use your GPU.

## 2. Test from another computer (optional, recommended)

Check the network path before involving the headset:

```bash
pip install websockets pillow numpy scikit-image   # + opencv-python for --webcam
python src/stream_client.py --url "ws://<IP>:8765/ws?token=pick-a-secret" --prompts "person,cup"
python src/stream_client.py --url "ws://<IP>:8765/ws?token=pick-a-secret" --webcam 0 --frames 100
```

It prints per-frame round-trip latency and FPS, and saves the last result
to `output/stream_client_last.png`.

## 3. Unity setup

Scripts are in [`unity/Sam3Stream/`](unity/Sam3Stream):

| Script | What it does |
|---|---|
| `Sam3StreamClient.cs` | WebSocket client. Sends frames from any `Texture`, receives masks, shows them on `RawImage`s, and reconnects automatically. |
| `QuestPassthroughCameraSource.cs` | Requests the headset-camera permission and feeds the live texture from Meta's `PassthroughCameraAccess` component to the client. |

1. Install Meta's **MR Utility Kit (MRUK)** package, which provides the
   `PassthroughCameraAccess` component.
2. Copy `unity/Sam3Stream/*.cs` into your project's `Assets/`.
3. Add a **world-space Canvas** in front of the user with two stacked,
   same-size `RawImage`s: `Preview` (the camera frame) and `Mask` on top.
4. Add `Sam3StreamClient` to an empty GameObject. Set `Server Url` to
   `ws://<IP>:8765/ws`, set `Token` and `Prompts`, and assign the two
   RawImages.
5. Add a **`PassthroughCameraAccess`** component to the scene. Then add
   `QuestPassthroughCameraSource` to the client's GameObject and assign
   both the `PassthroughCameraAccess` component and the client.
6. Android settings:
   - *Player > Other Settings > Internet Access = Require*.
   - If the connection is refused, also set *Allow downloads over HTTP =
     Always allowed*.
   - Add these permissions to `Assets/Plugins/Android/AndroidManifest.xml`:
     ```xml
     <uses-permission android:name="android.permission.CAMERA" />
     <uses-permission android:name="horizonos.permission.HEADSET_CAMERA" />
     ```
7. The Quest and the server must be on the same network, and that network
   must allow device-to-device traffic. Campus and guest Wi-Fi (e.g.
   eduroam) often isolates clients. If `http://<IP>:8765/` doesn't load in
   the Quest browser, use a router or hotspot you control.

Using it from your own code:

```csharp
client.SetPrompts("cup", "chair");          // change prompts at runtime
client.OnResult += result =>
{
    foreach (var d in result.detections)    // prompt, score, box, color
        Debug.Log($"{d.prompt} {d.score:F2} box={string.Join(",", d.box)}");
};
Texture2D mask = client.MaskTexture;       // latest RGBA overlay
```

Boxes are normalized `[x0, y0, x1, y1]` with the origin at the top-left.

## Troubleshooting: "connected, but no mask"

While frames flow, the server prints a status line every 5 seconds for each
client:

```
[stats] 167.96.152.6:46944 2.8 fps | frames 640x480 | brightness 6/255 | gpu 330 ms | dropped 0 | found: cup in 0/14 frames
  WARNING: frames are almost black - camera covered, headset not worn, or the client isn't reading the live camera texture
```

| What you see | Meaning |
|---|---|
| `[connect]` but no `[stats]` | Connected, but no frames are being sent. The camera source isn't ready: check `PassthroughCameraAccess` and the headset-camera permission. |
| `brightness` below ~20 + WARNING | Frames are black. Wear the headset (cameras uncovered), and make sure the client reads `cameraAccess.GetTexture()` live. |
| `found: cup in 0/N` with normal brightness | SAM 3 sees the scene but no match. Try another wording, or a lower `threshold` (e.g. 0.3). |
| `found: cup in N/N` but nothing in the headset | Unity display issue. Check that `Mask Image` is assigned and `Mask` is drawn on top of `Preview`. |

To see exactly what the headset sends, start the server with debug
snapshots. Every 5 seconds this writes `output/latest_frame.jpg` (the raw
frame received) and `output/latest_result.jpg` (the frame with the mask on
top):

```bash
SAM3_TOKEN=... SAM3_DEBUG_DIR=/app/output docker compose up
```

Snapshots contain your camera images, so they are off by default.

## Performance

Measured on an NVIDIA GB10 with 640px frames:

| Prompts | Per frame | FPS |
|---|---|---|
| 1 | ~330 ms | ~3.0 |
| 3 | ~430 ms | ~2.3 |

Almost all of that is SAM 3's image encoder, which always runs at
1008×1008 regardless of input size, so the network adds only a few ms.
Masks trail reality by about one inference, so keep the prompt count small.
The client keeps one frame in flight. If a client sends faster than that,
the server keeps only the newest frame, so latency never builds up.

## How it works

SAM 3's video predictor expects the whole video up front
(`start_session(resource_path=...)`), so it can't consume a live camera.
The server runs the SAM 3 **image** model on each incoming frame instead:
the image encoder runs once per frame, then the text-conditioned detection
head runs once per prompt. Frames are independent, so there are no object
IDs across frames.

All GPU work runs on one dedicated thread, which serializes access to the
single model when several clients connect.

## Protocol

Endpoint: `ws://<host>:8765/ws` (append `?token=...` if the server has one).
`GET /` is a JSON health check.

| Direction | Type | Payload |
|---|---|---|
| client → server | text | `{"type":"config","prompts":["cup"],"threshold":0.5}` |
| client → server | binary | 4-byte little-endian `uint32` frame id, then JPEG bytes |
| server → client | text | `{"type":"result","frame_id","width","height","inference_ms","brightness","detections":[{"prompt","prompt_index","score","box","color"}],"mask_png"}` |
| server → client | text | `ready`, `config_ack`, `error` (`{"type":"error","message",...}`) |

`mask_png` is a base64 RGBA PNG the same size as the frame. It is
transparent where nothing was found and tinted with each prompt's color
elsewhere. It is `null` when no prompts are configured.

## Project structure

```
sam3-unity/
├── src/
│   ├── stream_server.py   # WebSocket segmentation server (FastAPI + uvicorn)
│   └── stream_client.py   # Python test client (image or webcam)
├── unity/Sam3Stream/
│   ├── Sam3StreamClient.cs              # Unity WebSocket client
│   └── QuestPassthroughCameraSource.cs  # MRUK PassthroughCameraAccess -> client
├── examples/stream_result.png
├── Dockerfile             # fetches SAM 3's code at build time; no weights baked in
├── docker-compose.yml
├── .dockerignore / .gitignore
├── LICENSE                # MIT - covers ONLY this repo's own code, see NOTICE
└── README.md
```

## Limitations / next steps

- **Masks are shown on a panel, not anchored to the real world.**
  `PassthroughCameraAccess` already exposes what's needed to line them up
  with passthrough: `Intrinsics`, `GetCameraPose()` and
  `ViewportPointToRay()`.
- Around 2-3 FPS on a GB10. A faster GPU, fewer prompts or `torch.compile`
  are the main levers.
- `ReadPixels` + `EncodeToJPG` run on the main thread. `AsyncGPUReadback`
  would remove that stall on the headset.

## NOTICE - licensing of the model itself

This repository's own code is MIT-licensed - see [LICENSE](LICENSE).

**SAM 3 itself is not.** Meta's SAM 3 code and weights are "SAM Materials"
under Meta's **SAM License**
(https://github.com/facebookresearch/sam3/blob/main/LICENSE). Among other
things, it requires acknowledging SAM 3 in any publication of results and
prohibits certain military/surveillance uses. Neither the code nor the
weights are stored in this repository:

- The **Dockerfile clones Meta's official repo** at build time (pinned to
  commit `660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7`).
- The **checkpoint is downloaded at runtime** from Hugging Face with your
  own license-accepted account.

If you use SAM 3 in published work, please cite it (see the
[official README](https://github.com/facebookresearch/sam3#citing-sam-3)).

## Environment / versions used

| Component | Version |
|---|---|
| Server OS | Linux 6.17 (Ubuntu-based), ARM64/aarch64 |
| GPU | NVIDIA GB10 (Blackwell, sm_121), CUDA 13.0 |
| Docker base image | `python:3.12-slim` |
| torch / torchvision | 2.14.0+cu130 / 0.29.0+cu130 |
| SAM 3 source commit | `660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7` |
| SAM 3 checkpoint | `facebook/sam3` (848M params) |
| Headset | Meta Quest 3 / 3S, Horizon OS v74+ |

On a different GPU generation, change the PyTorch `--index-url` in the
Dockerfile (see https://pytorch.org/get-started/locally/).

## References

- N. Carion et al., "SAM 3: Segment Anything with Concepts," 2025.
  [arXiv:2511.16719](https://arxiv.org/abs/2511.16719)
- [Official SAM 3 repo](https://github.com/facebookresearch/sam3) ·
  [SAM 3 on Hugging Face](https://huggingface.co/facebook/sam3)
- [Meta Passthrough Camera API](https://developers.meta.com/horizon/documentation/unity/unity-pca-overview/)

## License

This repository's own code: MIT - see [LICENSE](LICENSE). SAM 3 itself is
licensed separately by Meta - see the NOTICE section above.
