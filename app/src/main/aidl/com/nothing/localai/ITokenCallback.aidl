package com.nothing.localai;

oneway interface ITokenCallback {
    void onToken(String requestId, String text);
    void onDone(String requestId, String fullText);
    void onError(String requestId, String code, String message);
}
