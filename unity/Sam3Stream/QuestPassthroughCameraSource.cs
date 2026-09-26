using System.Collections;
using UnityEngine;
using UnityEngine.Android;

/// Opens a Quest 3 / 3S passthrough camera (Horizon OS v74+) as a WebCamTexture and
/// feeds it to a Sam3StreamClient. In the Unity Editor it opens your PC webcam instead.
public class QuestPassthroughCameraSource : MonoBehaviour
{
    const string HeadsetCameraPermission = "horizonos.permission.HEADSET_CAMERA";

    public Sam3StreamClient client;
    [Tooltip("On Quest 3, 0 and 1 are the left and right passthrough cameras")]
    public int deviceIndex = 0;
    public int requestedWidth = 1280;
    public int requestedHeight = 960;

    public WebCamTexture Texture { get; private set; }

    IEnumerator Start()
    {
#if UNITY_ANDROID && !UNITY_EDITOR
        foreach (var permission in new[] { Permission.Camera, HeadsetCameraPermission })
        {
            if (Permission.HasUserAuthorizedPermission(permission)) continue;
            bool? granted = null;
            var callbacks = new PermissionCallbacks();
            callbacks.PermissionGranted += _ => granted = true;
            callbacks.PermissionDenied += _ => granted = false;
            Permission.RequestUserPermission(permission, callbacks);
            while (granted == null) yield return null;
            if (!granted.Value)
            {
                Debug.LogError($"[SAM3] permission {permission} denied - cannot access the passthrough camera");
                yield break;
            }
        }
#endif
        // The device list can take a moment to populate after the permission is granted.
        for (float t = 0; WebCamTexture.devices.Length == 0 && t < 5f; t += 0.25f)
            yield return new WaitForSeconds(0.25f);

        var devices = WebCamTexture.devices;
        if (devices.Length == 0)
        {
            Debug.LogError("[SAM3] no camera found (Quest 3/3S with Horizon OS v74+ required on device)");
            yield break;
        }

        var device = devices[Mathf.Clamp(deviceIndex, 0, devices.Length - 1)];
        Texture = new WebCamTexture(device.name, requestedWidth, requestedHeight, 30);
        Texture.Play();
        Debug.Log($"[SAM3] camera '{device.name}' started");
        if (client != null) client.source = Texture;
    }

    void OnDestroy()
    {
        if (Texture != null) Texture.Stop();
    }
}
