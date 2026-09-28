using System.Collections;
using Meta.XR;
using UnityEngine;
using UnityEngine.Android;

/// Feeds the Quest 3 / 3S passthrough camera (MRUK's PassthroughCameraAccess) to a Sam3StreamClient.
public class QuestPassthroughCameraSource : MonoBehaviour
{
    const string HeadsetCameraPermission = "horizonos.permission.HEADSET_CAMERA";

    [Header("Meta Quest Passthrough Camera")]
    public PassthroughCameraAccess cameraAccess;

    [Header("SAM3 Client")]
    public Sam3StreamClient client;

    bool loggedWaiting;

    IEnumerator Start()
    {
        if (cameraAccess == null) Debug.LogError("[SAM3] PassthroughCameraAccess is not assigned.");
        if (client == null) Debug.LogError("[SAM3] Sam3StreamClient is not assigned.");

#if UNITY_ANDROID && !UNITY_EDITOR
        // PassthroughCameraAccess only returns a valid texture once this permission is granted,
        // and it does not request it by itself.
        if (!Permission.HasUserAuthorizedPermission(HeadsetCameraPermission))
        {
            bool? granted = null;
            var callbacks = new PermissionCallbacks();
            callbacks.PermissionGranted += _ => granted = true;
            callbacks.PermissionDenied += _ => granted = false;
            Permission.RequestUserPermission(HeadsetCameraPermission, callbacks);
            while (granted == null) yield return null;
            if (!granted.Value) Debug.LogError("[SAM3] Headset camera permission denied.");
        }
#endif
        yield break;
    }

    void Update()
    {
        if (cameraAccess == null || client == null) return;

        if (!cameraAccess.IsPlaying)
        {
            if (!loggedWaiting) Debug.Log("[SAM3] Waiting for passthrough camera...");
            loggedWaiting = true;
            return;
        }

        // Re-read every frame: the component can hand out a new texture after a
        // pause/resume or resolution change, and a stale one streams as black.
        Texture cameraTexture = cameraAccess.GetTexture();
        if (cameraTexture == null || client.source == cameraTexture) return;

        client.source = cameraTexture;
        Debug.Log($"[SAM3] Passthrough camera connected: {cameraTexture.width} x {cameraTexture.height}");
    }
}
