/*
 * This file is auto-generated.  DO NOT MODIFY.
 * Using: /Users/patrickfan/Library/Android/sdk/build-tools/35.0.0/aidl -p/Users/patrickfan/Library/Android/sdk/platforms/android-36/framework.aidl -o/Users/patrickfan/Documents/GitHub/localai/app/build/generated/aidl_source_output_dir/debug/out -I/Users/patrickfan/Documents/GitHub/localai/app/src/main/aidl -I/Users/patrickfan/Documents/GitHub/localai/app/src/debug/aidl -I/Users/patrickfan/.gradle/caches/8.12/transforms/276be6965983aa840274d4510360d060/transformed/core-1.15.0/aidl -I/Users/patrickfan/.gradle/caches/8.12/transforms/d3471d695eaba187c52496bb0f37f125/transformed/versionedparcelable-1.1.1/aidl -d/var/folders/84/rn9qzcr573v3x4tpfjkktzwh0000gn/T/aidl2452548896973154384.d /Users/patrickfan/Documents/GitHub/localai/app/src/main/aidl/com/nothing/localai/IModelStatusCallback.aidl
 */
package com.nothing.localai;
public interface IModelStatusCallback extends android.os.IInterface
{
  /** Default implementation for IModelStatusCallback. */
  public static class Default implements com.nothing.localai.IModelStatusCallback
  {
    @Override public void onProgress(java.lang.String modelId, long bytesDownloaded, long totalBytes) throws android.os.RemoteException
    {
    }
    @Override public void onReady(java.lang.String modelId) throws android.os.RemoteException
    {
    }
    @Override public void onError(java.lang.String modelId, java.lang.String code, java.lang.String message) throws android.os.RemoteException
    {
    }
    @Override
    public android.os.IBinder asBinder() {
      return null;
    }
  }
  /** Local-side IPC implementation stub class. */
  public static abstract class Stub extends android.os.Binder implements com.nothing.localai.IModelStatusCallback
  {
    /** Construct the stub at attach it to the interface. */
    @SuppressWarnings("this-escape")
    public Stub()
    {
      this.attachInterface(this, DESCRIPTOR);
    }
    /**
     * Cast an IBinder object into an com.nothing.localai.IModelStatusCallback interface,
     * generating a proxy if needed.
     */
    public static com.nothing.localai.IModelStatusCallback asInterface(android.os.IBinder obj)
    {
      if ((obj==null)) {
        return null;
      }
      android.os.IInterface iin = obj.queryLocalInterface(DESCRIPTOR);
      if (((iin!=null)&&(iin instanceof com.nothing.localai.IModelStatusCallback))) {
        return ((com.nothing.localai.IModelStatusCallback)iin);
      }
      return new com.nothing.localai.IModelStatusCallback.Stub.Proxy(obj);
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
        case TRANSACTION_onProgress:
        {
          java.lang.String _arg0;
          _arg0 = data.readString();
          long _arg1;
          _arg1 = data.readLong();
          long _arg2;
          _arg2 = data.readLong();
          this.onProgress(_arg0, _arg1, _arg2);
          break;
        }
        case TRANSACTION_onReady:
        {
          java.lang.String _arg0;
          _arg0 = data.readString();
          this.onReady(_arg0);
          break;
        }
        case TRANSACTION_onError:
        {
          java.lang.String _arg0;
          _arg0 = data.readString();
          java.lang.String _arg1;
          _arg1 = data.readString();
          java.lang.String _arg2;
          _arg2 = data.readString();
          this.onError(_arg0, _arg1, _arg2);
          break;
        }
        default:
        {
          return super.onTransact(code, data, reply, flags);
        }
      }
      return true;
    }
    private static class Proxy implements com.nothing.localai.IModelStatusCallback
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
      @Override public void onProgress(java.lang.String modelId, long bytesDownloaded, long totalBytes) throws android.os.RemoteException
      {
        android.os.Parcel _data = android.os.Parcel.obtain();
        try {
          _data.writeInterfaceToken(DESCRIPTOR);
          _data.writeString(modelId);
          _data.writeLong(bytesDownloaded);
          _data.writeLong(totalBytes);
          boolean _status = mRemote.transact(Stub.TRANSACTION_onProgress, _data, null, android.os.IBinder.FLAG_ONEWAY);
        }
        finally {
          _data.recycle();
        }
      }
      @Override public void onReady(java.lang.String modelId) throws android.os.RemoteException
      {
        android.os.Parcel _data = android.os.Parcel.obtain();
        try {
          _data.writeInterfaceToken(DESCRIPTOR);
          _data.writeString(modelId);
          boolean _status = mRemote.transact(Stub.TRANSACTION_onReady, _data, null, android.os.IBinder.FLAG_ONEWAY);
        }
        finally {
          _data.recycle();
        }
      }
      @Override public void onError(java.lang.String modelId, java.lang.String code, java.lang.String message) throws android.os.RemoteException
      {
        android.os.Parcel _data = android.os.Parcel.obtain();
        try {
          _data.writeInterfaceToken(DESCRIPTOR);
          _data.writeString(modelId);
          _data.writeString(code);
          _data.writeString(message);
          boolean _status = mRemote.transact(Stub.TRANSACTION_onError, _data, null, android.os.IBinder.FLAG_ONEWAY);
        }
        finally {
          _data.recycle();
        }
      }
    }
    static final int TRANSACTION_onProgress = (android.os.IBinder.FIRST_CALL_TRANSACTION + 0);
    static final int TRANSACTION_onReady = (android.os.IBinder.FIRST_CALL_TRANSACTION + 1);
    static final int TRANSACTION_onError = (android.os.IBinder.FIRST_CALL_TRANSACTION + 2);
  }
  /** @hide */
  public static final java.lang.String DESCRIPTOR = "com.nothing.localai.IModelStatusCallback";
  public void onProgress(java.lang.String modelId, long bytesDownloaded, long totalBytes) throws android.os.RemoteException;
  public void onReady(java.lang.String modelId) throws android.os.RemoteException;
  public void onError(java.lang.String modelId, java.lang.String code, java.lang.String message) throws android.os.RemoteException;
}
