// ILocalAiService.aidl
// Contract between Aiwidget (host) and com.nothing.localai (model service).
// Mirrored byte-identical in the Aiwidget project. Methods may only be APPENDED
// at the end; reordering or removing breaks binary compatibility.
package com.nothing.localai;

import com.nothing.localai.ITokenCallback;
import com.nothing.localai.IModelStatusCallback;
import android.os.ParcelFileDescriptor;

interface ILocalAiService {

    // ===== Service =====
    int getApiVersion();

    // ===== Model lifecycle =====
    String getModelStatus(String modelId);
    void ensureModel(String modelId, IModelStatusCallback cb);

    // ===== LLM =====
    void createSession(String sessionId);
    void releaseSession(String sessionId);
    void resetSession(String sessionId);
    String generate(String sessionId, String prompt, ITokenCallback cb);
    void cancel(String requestId);

    // ===== Vision =====
    String classifyImage(in ParcelFileDescriptor jpegFd, int topK);

    // ===== Audio =====
    String transcribe(in ParcelFileDescriptor pcmFd, int sampleRate);
    void speak(String text);

    // ===== Multimodal LLM input (Gemma 3n) =====
    void addImage(String sessionId, in ParcelFileDescriptor jpegFd);
    void addAudio(String sessionId, in ParcelFileDescriptor pcmFd, int sampleRate);
}
