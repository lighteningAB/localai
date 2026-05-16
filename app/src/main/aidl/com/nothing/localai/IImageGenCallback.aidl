// IImageGenCallback.aidl
// Streaming callback for on-device image generation. Mirrored byte-identical in Aiwidget.
// `onStep` is optional — fires once per diffusion iteration with progress info; `onResult`
// delivers the final PNG via a ParcelFileDescriptor (read-only) so the encoded bytes never
// cross the binder boundary in-memory. The producer (LocalAiService) writes the PNG to its
// cacheDir and deletes it after the consumer closes the PFD or after a short TTL.
package com.nothing.localai;

import android.os.ParcelFileDescriptor;

oneway interface IImageGenCallback {
    void onStep(String requestId, int step, int totalSteps);
    void onResult(String requestId, in ParcelFileDescriptor pngFd, int width, int height);
    void onError(String requestId, String code, String message);

    // Outfit-swap stages: "segmenting" | "encoding" | "diffusing" | "decoding".
    // Fires once per stage transition. Append-only; existing consumers may ignore.
    void onStage(String requestId, String stageName);
}
