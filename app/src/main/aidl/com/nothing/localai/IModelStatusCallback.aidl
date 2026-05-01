package com.nothing.localai;

oneway interface IModelStatusCallback {
    void onProgress(String modelId, long bytesDownloaded, long totalBytes);
    void onReady(String modelId);
    void onError(String modelId, String code, String message);
}
