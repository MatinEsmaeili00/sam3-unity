using System;
using System.IO;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.UI;

[Serializable]
public class Sam3Detection
{
    public string prompt;
    public int prompt_index;
    public float score;
    public float[] box;   // [x0, y0, x1, y1], normalized 0..1, origin top-left
    public int[] color;   // [r, g, b] used for this prompt in the mask overlay
}

[Serializable]
public class Sam3Result
{
    public string type;
    public int frame_id;
    public int width;
    public int height;
    public float inference_ms;
    public Sam3Detection[] detections;
    public string mask_png;
    public string message;
}

/// Streams frames from `source` to the SAM 3 stream server and exposes the returned masks.
/// One frame is in flight at a time, so latency stays at a single inference round-trip.
public class Sam3StreamClient : MonoBehaviour
{
    [Header("Server")]
    public string serverUrl = "ws://192.168.1.50:8765/ws";
    public string token = "";

    [Header("Prompts")]
    public string[] prompts = { "cup" };
    [Range(0f, 1f)] public float threshold = 0.5f;

    [Header("Input")]
    public Texture source;
    public int sendWidth = 640;
    [Range(1, 100)] public int jpegQuality = 75;

    [Header("Output (optional)")]
    public RawImage previewImage;
    public RawImage maskImage;

    public event Action<Sam3Result> OnResult;
    public Sam3Result LatestResult { get; private set; }
    public Texture2D MaskTexture { get; private set; }
    public bool IsConnected { get; private set; }

    [Serializable]
    class ConfigMessage
    {
        public string type = "config";
        public string[] prompts;
        public float threshold;
    }

    CancellationTokenSource cts;
    RenderTexture scaled;
    Texture2D readback;
    Texture2D preview;
    int frameId;
    volatile bool configDirty = true;

    public void SetPrompts(params string[] newPrompts)
    {
        prompts = newPrompts;
        configDirty = true;
    }

    void OnValidate() => configDirty = true;

    async void Start()
    {
        cts = new CancellationTokenSource();
        MaskTexture = new Texture2D(2, 2, TextureFormat.RGBA32, false);
        while (!cts.IsCancellationRequested)
        {
            try
            {
                await StreamAsync(cts.Token);
            }
            catch (OperationCanceledException) { return; }
            catch (Exception e)
            {
                Debug.LogWarning($"[SAM3] {e.Message} - reconnecting in 2 s");
            }
            IsConnected = false;
            try { await Task.Delay(2000, cts.Token); } catch (OperationCanceledException) { return; }
        }
    }

    async Task StreamAsync(CancellationToken ct)
    {
        using var ws = new ClientWebSocket();
        var url = string.IsNullOrEmpty(token) ? serverUrl : $"{serverUrl}?token={Uri.EscapeDataString(token)}";
        await ws.ConnectAsync(new Uri(url), ct);
        IsConnected = true;
        configDirty = true;
        Debug.Log($"[SAM3] connected to {serverUrl}");

        while (ws.State == WebSocketState.Open)
        {
            if (configDirty)
            {
                configDirty = false;
                var json = JsonUtility.ToJson(new ConfigMessage { prompts = prompts, threshold = threshold });
                await ws.SendAsync(new ArraySegment<byte>(Encoding.UTF8.GetBytes(json)), WebSocketMessageType.Text, true, ct);
            }

            if (!SourceReady())
            {
                await Task.Delay(10, ct);
                continue;
            }

            int id = ++frameId;
            byte[] jpeg = CaptureJpeg();
            var packet = new byte[4 + jpeg.Length];
            BitConverter.GetBytes(id).CopyTo(packet, 0);  // little-endian on Quest (ARM) and PC
            jpeg.CopyTo(packet, 4);
            await ws.SendAsync(new ArraySegment<byte>(packet), WebSocketMessageType.Binary, true, ct);

            while (true)
            {
                var msg = await ReceiveAsync(ws, ct);
                if (msg == null) return;
                if (msg.type == "error") Debug.LogWarning($"[SAM3] server error: {msg.message}");
                if ((msg.type == "result" || msg.type == "error") && msg.frame_id == id)
                {
                    if (msg.type == "result") ApplyResult(msg);
                    break;
                }
            }
        }
    }

    bool SourceReady()
    {
        if (source == null || source.width <= 16) return false;  // WebCamTexture reports 16x16 until it starts
        return !(source is WebCamTexture cam) || cam.isPlaying;
    }

    byte[] CaptureJpeg()
    {
        int w = sendWidth;
        int h = Mathf.RoundToInt(sendWidth * (float)source.height / source.width);
        if (scaled == null || scaled.width != w || scaled.height != h)
        {
            if (scaled != null) scaled.Release();
            scaled = new RenderTexture(w, h, 0, RenderTextureFormat.ARGB32);
            readback = new Texture2D(w, h, TextureFormat.RGB24, false);
            preview = new Texture2D(w, h, TextureFormat.RGB24, false);
        }
        Graphics.Blit(source, scaled);
        var prev = RenderTexture.active;
        RenderTexture.active = scaled;
        readback.ReadPixels(new Rect(0, 0, w, h), 0, 0);
        readback.Apply(false);
        RenderTexture.active = prev;
        return readback.EncodeToJPG(jpegQuality);
    }

    void ApplyResult(Sam3Result result)
    {
        LatestResult = result;
        if (!string.IsNullOrEmpty(result.mask_png))
            MaskTexture.LoadImage(Convert.FromBase64String(result.mask_png));

        if (previewImage != null)
        {
            // Show the exact frame the mask was computed on, so the two stay aligned.
            Graphics.CopyTexture(readback, preview);
            previewImage.texture = preview;
        }
        if (maskImage != null)
        {
            maskImage.texture = MaskTexture;
            maskImage.enabled = !string.IsNullOrEmpty(result.mask_png);
        }
        OnResult?.Invoke(result);
    }

    static async Task<Sam3Result> ReceiveAsync(ClientWebSocket ws, CancellationToken ct)
    {
        var buffer = new ArraySegment<byte>(new byte[64 * 1024]);
        using var stream = new MemoryStream();
        WebSocketReceiveResult r;
        do
        {
            r = await ws.ReceiveAsync(buffer, ct);
            if (r.MessageType == WebSocketMessageType.Close) return null;
            stream.Write(buffer.Array, 0, r.Count);
        } while (!r.EndOfMessage);
        return JsonUtility.FromJson<Sam3Result>(Encoding.UTF8.GetString(stream.GetBuffer(), 0, (int)stream.Length));
    }

    void OnDestroy()
    {
        cts?.Cancel();
        if (scaled != null) scaled.Release();
    }
}
