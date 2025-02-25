package camp.visual.camera;

import android.content.Context;
import android.graphics.PixelFormat;
import android.graphics.SurfaceTexture;
import android.hardware.Camera;
import android.opengl.GLES20;
import android.opengl.GLSurfaceView;
import android.view.SurfaceHolder;
import android.view.SurfaceView;

import java.util.ArrayList;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

import camp.visual.ac.Images;
import camp.visual.kappa.display.GLRender;
import camp.visual.kappa.glutils.OpenGLUtils;

class CameraCalibrationDisplay implements GLSurfaceView.Renderer {
    private String TAG = "CameraCalibrationDisplay";

    private int mImageWidth;
    private int mImageHeight;
    private int mFrameSize;
    private byte[] mImageData;
    private int mCurrentPreviewId = 0;
    private boolean mCameraChanging = false;
    private ArrayList<String> mSupportedPreviewSizes;
    private final Object mImageDataLock = new Object();

    // Status
    private int mCaptureIdx = 0;
    private boolean mIsPaused;

    private CameraProxy mCameraProxy;
    private SurfaceHolder mSurfaceHolder;
    private Camera.PreviewCallback mPreviewCallback;

    // GL
    private int mSurfaceWidth;
    private int mSurfaceHeight;
    private GLRender mGLRender;
    private GLSurfaceView mGlSurfaceView;
    private SurfaceTexture mSurfaceTexture;
    private int mTextureId = OpenGLUtils.NO_TEXTURE;


    /*
     * init
     */
    CameraCalibrationDisplay(Context context, GLSurfaceView glView, SurfaceView surfaceView) {
        mGLRender = new GLRender();
        mCameraProxy = new CameraProxy(context);

        setRenderGlView(glView);

        surfaceView.setZOrderMediaOverlay(true);
        mSurfaceHolder = surfaceView.getHolder();
        mSurfaceHolder.setFormat(PixelFormat.TRANSLUCENT);

        setupCallback();
    }

    private void setUpCamera() {
        if (mTextureId == OpenGLUtils.NO_TEXTURE) {
            mTextureId = OpenGLUtils.getExternalOESTextureID();
            mSurfaceTexture = new SurfaceTexture(mTextureId);
        }

        String size = mSupportedPreviewSizes.get(mCurrentPreviewId);
        int index = size.indexOf('x');
        mImageWidth = Integer.parseInt(size.substring(index + 1));
        mImageHeight = Integer.parseInt(size.substring(0, index));
        mFrameSize = mImageHeight * mImageWidth * 3 / 2;

        boolean flipHorizontal = mCameraProxy.isFlipHorizontal();
        mGLRender.adjustTextureBuffer(mCameraProxy.getOrientation(), flipHorizontal);

        mCameraProxy.setRotation(90);
        mCameraProxy.setPreviewSize(mImageHeight, mImageWidth);
        mCameraProxy.startPreview(mSurfaceTexture, mPreviewCallback);
    }

    private void setupCallback() {
        mPreviewCallback = (data, camera) -> {
            if (mCameraChanging || mCameraProxy.getCamera() == null) {
                return;
            }

            if (mImageData == null || mImageData.length != mFrameSize) {
                mImageData = new byte[mFrameSize];
            }

            synchronized (mImageDataLock) {
                System.arraycopy(data, 0, mImageData, 0, data.length);
            }

            mGlSurfaceView.requestRender();
        };
    }

    private void setRenderGlView(GLSurfaceView glView) {
        mGlSurfaceView = glView;
        mGlSurfaceView.setEGLContextClientVersion(2);
        mGlSurfaceView.setRenderer(this);
        mGlSurfaceView.setRenderMode(GLSurfaceView.RENDERMODE_WHEN_DIRTY);
    }


    /*
     * API
     */
    void capture() {
        byte[] tmpBuffer = new byte[mFrameSize];

        synchronized (mImageDataLock) {
            System.arraycopy(mImageData, 0, tmpBuffer, 0, mImageData.length);
            mCaptureIdx++;
        }

        Images.saveImage(tmpBuffer, mImageHeight, mImageWidth, String.valueOf(mCaptureIdx));
    }



    /*
     * Life Cycle and GL
     */
    void onResume() {
        mIsPaused = false;

        if (mCameraProxy.getCamera() == null) {
            int mCameraID = Camera.CameraInfo.CAMERA_FACING_FRONT;
            mCameraProxy.openCamera(mCameraID);
            mSupportedPreviewSizes = mCameraProxy.getSupportedPreviewSize(
                    new String[] {"1280x720", "640x480" });
        }

        mGLRender = new GLRender();

        mGlSurfaceView.onResume();
        mGlSurfaceView.forceLayout();
        mGlSurfaceView.requestRender();
    }

    void onPause() {
        mIsPaused = true;

        mGlSurfaceView.queueEvent(() -> {
            deleteTextures();
            if (mSurfaceTexture != null) {
                mSurfaceTexture.release();
            }
            mGLRender.destroyFrameBuffers();
        });

        mGlSurfaceView.onPause();
        mCameraProxy.releaseCamera();
    }

    void onDestroy() {
        mImageData = null;
    }

    void changePreviewSize(int currentPreview) {
        if (mCameraProxy.getCamera() == null || mCameraChanging || mIsPaused) {
            return;
        }

        mCurrentPreviewId = currentPreview;
        mCameraChanging = true;
        mCameraProxy.stopPreview();

        mGlSurfaceView.queueEvent(() -> {
            deleteTextures();
            if (mCameraProxy.getCamera() != null) {
                setUpCamera();
            }

            mGLRender.destroySegmentFrameBuffer();
            mGLRender.init(mImageWidth, mImageHeight);
            mGLRender.initDrawBackGround();
            mGLRender.calculateVertexBuffer(mSurfaceWidth, mSurfaceHeight, mImageWidth, mImageHeight);

            mCameraChanging = false;
        });
    }

    private void deleteTextures() {
        if (mTextureId != OpenGLUtils.NO_TEXTURE) {
            GLES20.glDeleteTextures(1, new int[]{mTextureId}, 0);
        }

        mTextureId = OpenGLUtils.NO_TEXTURE;
        mGLRender.deleteSegmentResultTexture();
    }

    @Override
    public void onSurfaceCreated(GL10 gl, EGLConfig config) {
        if (mIsPaused) {
            return;
        }

        GLES20.glEnable(GL10.GL_DITHER);
        GLES20.glClearColor(0, 0, 0, 0);
        GLES20.glEnable(GL10.GL_DEPTH_TEST);

        if (mCameraProxy.getCamera() != null) {
            setUpCamera();
        }
    }

    @Override
    public void onSurfaceChanged(GL10 gl, int width, int height) {
        adjustViewPort(width, height);
        mGLRender.init(mImageWidth, mImageHeight);
    }

    private void adjustViewPort(int width, int height) {
        mSurfaceHeight = height;
        mSurfaceWidth = width;
        GLES20.glViewport(0, 0, mSurfaceWidth, mSurfaceHeight);
        mGLRender.calculateVertexBuffer(mSurfaceWidth, mSurfaceHeight, mImageWidth, mImageHeight);
    }

    @Override
    public void onDrawFrame(GL10 gl) {
        if (mCameraChanging || mCameraProxy.getCamera() == null || mSurfaceTexture == null) {
            return;
        }

        mSurfaceTexture.updateTexImage();
        GLES20.glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT | GLES20.GL_DEPTH_BUFFER_BIT);

        int textureId = mGLRender.preProcess(mTextureId, null);

        GLES20.glViewport(0, 0, mSurfaceWidth, mSurfaceHeight);
        mGLRender.onDrawFrame(textureId);
    }

}