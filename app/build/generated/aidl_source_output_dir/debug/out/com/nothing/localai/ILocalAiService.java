/*
 * This file is auto-generated.  DO NOT MODIFY.
 * Using: /Users/patrickfan/Library/Android/sdk/build-tools/35.0.0/aidl -p/Users/patrickfan/Library/Android/sdk/platforms/android-36/framework.aidl -o/Users/patrickfan/Documents/GitHub/localai/app/build/generated/aidl_source_output_dir/debug/out -I/Users/patrickfan/Documents/GitHub/localai/app/src/main/aidl -I/Users/patrickfan/Documents/GitHub/localai/app/src/debug/aidl -I/Users/patrickfan/.gradle/caches/8.12/transforms/276be6965983aa840274d4510360d060/transformed/core-1.15.0/aidl -I/Users/patrickfan/.gradle/caches/8.12/transforms/d3471d695eaba187c52496bb0f37f125/transformed/versionedparcelable-1.1.1/aidl -d/var/folders/84/rn9qzcr573v3x4tpfjkktzwh0000gn/T/aidl16430808525756712673.d /Users/patrickfan/Documents/GitHub/localai/app/src/main/aidl/com/nothing/localai/ILocalAiService.aidl
 */
package com.nothing.localai;
public interface ILocalAiService extends android.os.IInterface
{
  /** Default implementation for ILocalAiService. */
  public static class Default implements com.nothing.localai.ILocalAiService
  {
    // ===== Service =====
    @Override public int getApiVersion() throws android.os.RemoteException
    {
      return 0;
    }
    // ===== Model lifecycle =====
    @Override public java.lang.String getModelStatus(java.lang.String modelId) throws android.os.RemoteException
    {
      return null;
    }
    @Override public void ensureModel(java.lang.String modelId, com.nothing.localai.IModelStatusCallback cb) throws android.os.RemoteException
    {
    }
    // ===== LLM =====
    @Override public void createSession(java.lang.String sessionId) throws android.os.RemoteException
    {
    }
    @Override public void releaseSession(java.lang.String sessionId) throws android.os.RemoteException
    {
    }
    @Override public void resetSession(java.lang.String sessionId) throws android.os.RemoteException
    {
    }
    @Override public java.lang.String generate(java.lang.String sessionId, java.lang.String prompt, com.nothing.localai.ITokenCallback cb) throws android.os.RemoteException
    {
      return null;
    }
    @Override public void cancel(java.lang.String requestId) throws android.os.RemoteException
    {
    }
    // ===== Vision =====
    @Override public java.lang.String classifyImage(android.os.ParcelFileDescriptor jpegFd, int topK) throws android.os.RemoteException
    {
      return null;
    }
    // ===== Audio =====
    @Override public java.lang.String transcribe(android.os.ParcelFileDescriptor pcmFd, int sampleRate) throws android.os.RemoteException
    {
      return null;
    }
    @Override public void speak(java.lang.String text) throws android.os.RemoteException
    {
    }
    // ===== Multimodal LLM input (Gemma 3n) =====
    @Override public void addImage(java.lang.String sessionId, android.os.ParcelFileDescriptor jpegFd) throws android.os.RemoteException
    {
    }
    @Override public void addAudio(java.lang.String sessionId, android.os.ParcelFileDescriptor pcmFd, int sampleRate) throws android.os.RemoteException
    {
    }
    @Override
    public android.os.IBinder asBinder() {
      return null;
    }
  }
  /** Local-side IPC implementation stub class. */
  public static abstract class Stub extends android.os.Binder implements com.nothing.localai.ILocalAiService
  {
    /** Construct the stub at attach it to the interface. */
    @SuppressWarnings("this-escape")
    public Stub()
    {
      this.attachInterface(this, DESCRIPTOR);
    }
    /**
     * Cast an IBinder object into an com.nothing.localai.ILocalAiService interface,
     * generating a proxy if needed.
     */
    public static com.nothing.localai.ILocalAiService asInterface(android.os.IBinder obj)
    {
      if ((obj==null)) {
        return null;
      }
      android.os.IInterface iin = obj.queryLocalInterface(DESCRIPTOR);
      if (((iin!=null)&&(iin instanceof com.nothing.localai.ILocalAiService))) {
        return ((com.nothing.localai.ILocalAiService)iin);
      }
      return new com.nothing.localai.ILocalAiService.Stub.Proxy(obj);
    }
    @Override public android.os.IBinder asBinder()
    {
      return this;
    }
    @Override public boolean onTransact(int code, android.os.Parcel data, android.os.Parcel reply, int flags) throws android.os.RemoteException
    {
      java.lang.String descriptor = DESCRIPTOR;
      if (code >= android.os.IBinder.FIRST_CALL_TRANSACTION && code <= android.os.IBinder.LAST_CALL_TRANSACTION) {
        data.enforceInterface(descriptor);
      }
      if (code == INTERFACE_TRANSACTION) {
        reply.writeString(descriptor);
        return true;
      }
      switch (code)
      {
        case TRANSACTION_getApiVersion:
        {
          int _result = this.getApiVersion();
          reply.writeNoException();
          reply.writeInt(_result);
          break;
        }
        case TRANSACTION_getModelStatus:
        {
          java.lang.String _arg0;
          _arg0 = data.readString();
          java.lang.String _result = this.getModelStatus(_arg0);
          reply.writeNoException();
          reply.writeString(_result);
          break;
        }
        case TRANSACTION_ensureModel:
        {
          java.lang.String _arg0;
          _arg0 = data.readString();
          com.nothing.localai.IModelStatusCallback _arg1;
          _arg1 = com.nothing.localai.IModelStatusCallback.Stub.asInterface(data.readStrongBinder());
          this.ensureModel(_arg0, _arg1);
          reply.writeNoException();
          break;
        }
        case TRANSACTION_createSession:
        {
          java.lang.String _arg0;
          _arg0 = data.readString();
          this.createSession(_arg0);
          reply.writeNoException();
          break;
        }
        case TRANSACTION_releaseSession:
        {
          java.lang.String _arg0;
          _arg0 = data.readString();
          this.releaseSession(_arg0);
          reply.writeNoException();
          break;
        }
        case TRANSACTION_resetSession:
        {
          java.lang.String _arg0;
          _arg0 = data.readString();
          this.resetSession(_arg0);
          reply.writeNoException();
          break;
        }
        case TRANSACTION_generate:
        {
          java.lang.String _arg0;
          _arg0 = data.readString();
          java.lang.String _arg1;
          _arg1 = data.readString();
          com.nothing.localai.ITokenCallback _arg2;
          _arg2 = com.nothing.localai.ITokenCallback.Stub.asInterface(data.readStrongBinder());
          java.lang.String _result = this.generate(_arg0, _arg1, _arg2);
          reply.writeNoException();
          reply.writeString(_result);
          break;
        }
        case TRANSACTION_cancel:
        {
          java.lang.String _arg0;
          _arg0 = data.readString();
          this.cancel(_arg0);
          reply.writeNoException();
          break;
        }
        case TRANSACTION_classifyImage:
        {
          android.os.ParcelFileDescriptor _arg0;
          _arg0 = _Parcel.readTypedObject(data, android.os.ParcelFileDescriptor.CREATOR);
          int _arg1;
          _arg1 = data.readInt();
          java.lang.String _result = this.classifyImage(_arg0, _arg1);
          reply.writeNoException();
          reply.writeString(_result);
          break;
        }
        case TRANSACTION_transcribe:
        {
          android.os.ParcelFileDescriptor _arg0;
          _arg0 = _Parcel.readTypedObject(data, android.os.ParcelFileDescriptor.CREATOR);
          int _arg1;
          _arg1 = data.readInt();
          java.lang.String _result = this.transcribe(_arg0, _arg1);
          reply.writeNoException();
          reply.writeString(_result);
          break;
        }
        case TRANSACTION_speak:
        {
          java.lang.String _arg0;
          _arg0 = data.readString();
          this.speak(_arg0);
          reply.writeNoException();
          break;
        }
        case TRANSACTION_addImage:
        {
          java.lang.String _arg0;
          _arg0 = data.readString();
          android.os.ParcelFileDescriptor _arg1;
          _arg1 = _Parcel.readTypedObject(data, android.os.ParcelFileDescriptor.CREATOR);
          this.addImage(_arg0, _arg1);
          reply.writeNoException();
          break;
        }
        case TRANSACTION_addAudio:
        {
          java.lang.String _arg0;
          _arg0 = data.readString();
          android.os.ParcelFileDescriptor _arg1;
          _arg1 = _Parcel.readTypedObject(data, android.os.ParcelFileDescriptor.CREATOR);
          int _arg2;
          _arg2 = data.readInt();
          this.addAudio(_arg0, _arg1, _arg2);
          reply.writeNoException();
          break;
        }
        default:
        {
          return super.onTransact(code, data, reply, flags);
        }
      }
      return true;
    }
    private static class Proxy implements com.nothing.localai.ILocalAiService
    {
      private android.os.IBinder mRemote;
      Proxy(android.os.IBinder remote)
      {
        mRemote = remote;
      }
      @Override public android.os.IBinder asBinder()
      {
        return mRemote;
      }
      public java.lang.String getInterfaceDescriptor()
      {
        return DESCRIPTOR;
      }
      // ===== Service =====
      @Override public int getApiVersion() throws android.os.RemoteException
      {
        android.os.Parcel _data = android.os.Parcel.obtain();
        android.os.Parcel _reply = android.os.Parcel.obtain();
        int _result;
        try {
          _data.writeInterfaceToken(DESCRIPTOR);
          boolean _status = mRemote.transact(Stub.TRANSACTION_getApiVersion, _data, _reply, 0);
          _reply.readException();
          _result = _reply.readInt();
        }
        finally {
          _reply.recycle();
          _data.recycle();
        }
        return _result;
      }
      // ===== Model lifecycle =====
      @Override public java.lang.String getModelStatus(java.lang.String modelId) throws android.os.RemoteException
      {
        android.os.Parcel _data = android.os.Parcel.obtain();
        android.os.Parcel _reply = android.os.Parcel.obtain();
        java.lang.String _result;
        try {
          _data.writeInterfaceToken(DESCRIPTOR);
          _data.writeString(modelId);
          boolean _status = mRemote.transact(Stub.TRANSACTION_getModelStatus, _data, _reply, 0);
          _reply.readException();
          _result = _reply.readString();
        }
        finally {
          _reply.recycle();
          _data.recycle();
        }
        return _result;
      }
      @Override public void ensureModel(java.lang.String modelId, com.nothing.localai.IModelStatusCallback cb) throws android.os.RemoteException
      {
        android.os.Parcel _data = android.os.Parcel.obtain();
        android.os.Parcel _reply = android.os.Parcel.obtain();
        try {
          _data.writeInterfaceToken(DESCRIPTOR);
          _data.writeString(modelId);
          _data.writeStrongInterface(cb);
          boolean _status = mRemote.transact(Stub.TRANSACTION_ensureModel, _data, _reply, 0);
          _reply.readException();
        }
        finally {
          _reply.recycle();
          _data.recycle();
        }
      }
      // ===== LLM =====
      @Override public void createSession(java.lang.String sessionId) throws android.os.RemoteException
      {
        android.os.Parcel _data = android.os.Parcel.obtain();
        android.os.Parcel _reply = android.os.Parcel.obtain();
        try {
          _data.writeInterfaceToken(DESCRIPTOR);
          _data.writeString(sessionId);
          boolean _status = mRemote.transact(Stub.TRANSACTION_createSession, _data, _reply, 0);
          _reply.readException();
        }
        finally {
          _reply.recycle();
          _data.recycle();
        }
      }
      @Override public void releaseSession(java.lang.String sessionId) throws android.os.RemoteException
      {
        android.os.Parcel _data = android.os.Parcel.obtain();
        android.os.Parcel _reply = android.os.Parcel.obtain();
        try {
          _data.writeInterfaceToken(DESCRIPTOR);
          _data.writeString(sessionId);
          boolean _status = mRemote.transact(Stub.TRANSACTION_releaseSession, _data, _reply, 0);
          _reply.readException();
        }
        finally {
          _reply.recycle();
          _data.recycle();
        }
      }
      @Override public void resetSession(java.lang.String sessionId) throws android.os.RemoteException
      {
        android.os.Parcel _data = android.os.Parcel.obtain();
        android.os.Parcel _reply = android.os.Parcel.obtain();
        try {
          _data.writeInterfaceToken(DESCRIPTOR);
          _data.writeString(sessionId);
          boolean _status = mRemote.transact(Stub.TRANSACTION_resetSession, _data, _reply, 0);
          _reply.readException();
        }
        finally {
          _reply.recycle();
          _data.recycle();
        }
      }
      @Override public java.lang.String generate(java.lang.String sessionId, java.lang.String prompt, com.nothing.localai.ITokenCallback cb) throws android.os.RemoteException
      {
        android.os.Parcel _data = android.os.Parcel.obtain();
        android.os.Parcel _reply = android.os.Parcel.obtain();
        java.lang.String _result;
        try {
          _data.writeInterfaceToken(DESCRIPTOR);
          _data.writeString(sessionId);
          _data.writeString(prompt);
          _data.writeStrongInterface(cb);
          boolean _status = mRemote.transact(Stub.TRANSACTION_generate, _data, _reply, 0);
          _reply.readException();
          _result = _reply.readString();
        }
        finally {
          _reply.recycle();
          _data.recycle();
        }
        return _result;
      }
      @Override public void cancel(java.lang.String requestId) throws android.os.RemoteException
      {
        android.os.Parcel _data = android.os.Parcel.obtain();
        android.os.Parcel _reply = android.os.Parcel.obtain();
        try {
          _data.writeInterfaceToken(DESCRIPTOR);
          _data.writeString(requestId);
          boolean _status = mRemote.transact(Stub.TRANSACTION_cancel, _data, _reply, 0);
          _reply.readException();
        }
        finally {
          _reply.recycle();
          _data.recycle();
        }
      }
      // ===== Vision =====
      @Override public java.lang.String classifyImage(android.os.ParcelFileDescriptor jpegFd, int topK) throws android.os.RemoteException
      {
        android.os.Parcel _data = android.os.Parcel.obtain();
        android.os.Parcel _reply = android.os.Parcel.obtain();
        java.lang.String _result;
        try {
          _data.writeInterfaceToken(DESCRIPTOR);
          _Parcel.writeTypedObject(_data, jpegFd, 0);
          _data.writeInt(topK);
          boolean _status = mRemote.transact(Stub.TRANSACTION_classifyImage, _data, _reply, 0);
          _reply.readException();
          _result = _reply.readString();
        }
        finally {
          _reply.recycle();
          _data.recycle();
        }
        return _result;
      }
      // ===== Audio =====
      @Override public java.lang.String transcribe(android.os.ParcelFileDescriptor pcmFd, int sampleRate) throws android.os.RemoteException
      {
        android.os.Parcel _data = android.os.Parcel.obtain();
        android.os.Parcel _reply = android.os.Parcel.obtain();
        java.lang.String _result;
        try {
          _data.writeInterfaceToken(DESCRIPTOR);
          _Parcel.writeTypedObject(_data, pcmFd, 0);
          _data.writeInt(sampleRate);
          boolean _status = mRemote.transact(Stub.TRANSACTION_transcribe, _data, _reply, 0);
          _reply.readException();
          _result = _reply.readString();
        }
        finally {
          _reply.recycle();
          _data.recycle();
        }
        return _result;
      }
      @Override public void speak(java.lang.String text) throws android.os.RemoteException
      {
        android.os.Parcel _data = android.os.Parcel.obtain();
        android.os.Parcel _reply = android.os.Parcel.obtain();
        try {
          _data.writeInterfaceToken(DESCRIPTOR);
          _data.writeString(text);
          boolean _status = mRemote.transact(Stub.TRANSACTION_speak, _data, _reply, 0);
          _reply.readException();
        }
        finally {
          _reply.recycle();
          _data.recycle();
        }
      }
      // ===== Multimodal LLM input (Gemma 3n) =====
      @Override public void addImage(java.lang.String sessionId, android.os.ParcelFileDescriptor jpegFd) throws android.os.RemoteException
      {
        android.os.Parcel _data = android.os.Parcel.obtain();
        android.os.Parcel _reply = android.os.Parcel.obtain();
        try {
          _data.writeInterfaceToken(DESCRIPTOR);
          _data.writeString(sessionId);
          _Parcel.writeTypedObject(_data, jpegFd, 0);
          boolean _status = mRemote.transact(Stub.TRANSACTION_addImage, _data, _reply, 0);
          _reply.readException();
        }
        finally {
          _reply.recycle();
          _data.recycle();
        }
      }
      @Override public void addAudio(java.lang.String sessionId, android.os.ParcelFileDescriptor pcmFd, int sampleRate) throws android.os.RemoteException
      {
        android.os.Parcel _data = android.os.Parcel.obtain();
        android.os.Parcel _reply = android.os.Parcel.obtain();
        try {
          _data.writeInterfaceToken(DESCRIPTOR);
          _data.writeString(sessionId);
          _Parcel.writeTypedObject(_data, pcmFd, 0);
          _data.writeInt(sampleRate);
          boolean _status = mRemote.transact(Stub.TRANSACTION_addAudio, _data, _reply, 0);
          _reply.readException();
        }
        finally {
          _reply.recycle();
          _data.recycle();
        }
      }
    }
    static final int TRANSACTION_getApiVersion = (android.os.IBinder.FIRST_CALL_TRANSACTION + 0);
    static final int TRANSACTION_getModelStatus = (android.os.IBinder.FIRST_CALL_TRANSACTION + 1);
    static final int TRANSACTION_ensureModel = (android.os.IBinder.FIRST_CALL_TRANSACTION + 2);
    static final int TRANSACTION_createSession = (android.os.IBinder.FIRST_CALL_TRANSACTION + 3);
    static final int TRANSACTION_releaseSession = (android.os.IBinder.FIRST_CALL_TRANSACTION + 4);
    static final int TRANSACTION_resetSession = (android.os.IBinder.FIRST_CALL_TRANSACTION + 5);
    static final int TRANSACTION_generate = (android.os.IBinder.FIRST_CALL_TRANSACTION + 6);
    static final int TRANSACTION_cancel = (android.os.IBinder.FIRST_CALL_TRANSACTION + 7);
    static final int TRANSACTION_classifyImage = (android.os.IBinder.FIRST_CALL_TRANSACTION + 8);
    static final int TRANSACTION_transcribe = (android.os.IBinder.FIRST_CALL_TRANSACTION + 9);
    static final int TRANSACTION_speak = (android.os.IBinder.FIRST_CALL_TRANSACTION + 10);
    static final int TRANSACTION_addImage = (android.os.IBinder.FIRST_CALL_TRANSACTION + 11);
    static final int TRANSACTION_addAudio = (android.os.IBinder.FIRST_CALL_TRANSACTION + 12);
  }
  /** @hide */
  public static final java.lang.String DESCRIPTOR = "com.nothing.localai.ILocalAiService";
  // ===== Service =====
  public int getApiVersion() throws android.os.RemoteException;
  // ===== Model lifecycle =====
  public java.lang.String getModelStatus(java.lang.String modelId) throws android.os.RemoteException;
  public void ensureModel(java.lang.String modelId, com.nothing.localai.IModelStatusCallback cb) throws android.os.RemoteException;
  // ===== LLM =====
  public void createSession(java.lang.String sessionId) throws android.os.RemoteException;
  public void releaseSession(java.lang.String sessionId) throws android.os.RemoteException;
  public void resetSession(java.lang.String sessionId) throws android.os.RemoteException;
  public java.lang.String generate(java.lang.String sessionId, java.lang.String prompt, com.nothing.localai.ITokenCallback cb) throws android.os.RemoteException;
  public void cancel(java.lang.String requestId) throws android.os.RemoteException;
  // ===== Vision =====
  public java.lang.String classifyImage(android.os.ParcelFileDescriptor jpegFd, int topK) throws android.os.RemoteException;
  // ===== Audio =====
  public java.lang.String transcribe(android.os.ParcelFileDescriptor pcmFd, int sampleRate) throws android.os.RemoteException;
  public void speak(java.lang.String text) throws android.os.RemoteException;
  // ===== Multimodal LLM input (Gemma 3n) =====
  public void addImage(java.lang.String sessionId, android.os.ParcelFileDescriptor jpegFd) throws android.os.RemoteException;
  public void addAudio(java.lang.String sessionId, android.os.ParcelFileDescriptor pcmFd, int sampleRate) throws android.os.RemoteException;
  /** @hide */
  static class _Parcel {
    static private <T> T readTypedObject(
        android.os.Parcel parcel,
        android.os.Parcelable.Creator<T> c) {
      if (parcel.readInt() != 0) {
          return c.createFromParcel(parcel);
      } else {
          return null;
      }
    }
    static private <T extends android.os.Parcelable> void writeTypedObject(
        android.os.Parcel parcel, T value, int parcelableFlags) {
      if (value != null) {
        parcel.writeInt(1);
        value.writeToParcel(parcel, parcelableFlags);
      } else {
        parcel.writeInt(0);
      }
    }
  }
}
