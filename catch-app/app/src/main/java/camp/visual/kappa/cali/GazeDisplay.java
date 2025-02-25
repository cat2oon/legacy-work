package camp.visual.kappa.cali;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.PixelFormat;
import android.graphics.PointF;
import android.graphics.PorterDuff;
import android.graphics.Rect;
import android.graphics.SurfaceTexture;
import android.hardware.Camera;
import android.opengl.GLES20;
import android.opengl.GLSurfaceView;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Message;
import android.util.Log;
import android.view.SurfaceView;

import com.sensetime.stmobile.FaceDetectorParamsType;
import com.sensetime.stmobile.STCommonNative;
import com.sensetime.stmobile.STMobileHumanActionNative;
import com.sensetime.stmobile.STRotateType;
import com.sensetime.stmobile.model.STHumanAction;
import com.sensetime.stmobile.model.STMobileFaceInfo;
import com.sensetime.stmobile.model.STPoint;

import org.apache.commons.lang3.exception.ExceptionUtils;
import org.eclipse.collections.impl.map.mutable.UnifiedMap;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

import camp.visual.kappa.al.SubjectProxy;
import camp.visual.camera.CameraProxy;
import camp.visual.camera.ExtrinsicParam;
import camp.visual.kappa.display.GLRender;
import camp.visual.kappa.glutils.OpenGLUtils;
import camp.visual.kappa.utils.Accelerometer;
import camp.visual.kappa.utils.DetectorUtils;
import camp.visual.kappa.utils.FileUtils;
import camp.visual.kappa.utils.LogUtils;
import camp.visual.kappa.zt.ZTDetectionParser;

import static camp.visual.ac.Maths.toZeroMatrix;
import static org.apache.commons.math3.util.Precision.round;

class GazeDisplay implements GLSurfaceView.Renderer  {
    private String TAG = "GazeDisplay";

    private static final int MESSAGE_PROCESS_IMAGE = 100;

    private static final int CAPTURE_FORMAT = STCommonNative.ST_PIX_FMT_NV21;

    private int mFrameSize;
    private int mImageWidth;
    private int mImageHeight;
    private byte[] mImageData;
    private byte[] mTmpBuffer;
    private SurfaceTexture mSurfaceTexture;
    private int mTextureId = OpenGLUtils.NO_TEXTURE;
    private final Object mImageDataLock = new Object();

    private Context mContext;
    private CameraProxy mCameraProxy;
    private int mCameraID = Camera.CameraInfo.CAMERA_FACING_FRONT;

    private GLRender mGLRender;
    private GLSurfaceView mGlSurfaceView;

    private int mCurrentPreview = 0;
    private int mSurfaceWidth, mSurfaceHeight;
    private ArrayList<String> mSupportedPreviewSizes;


    /*
     * status
     */
    private boolean mIsPaused = false;
    private boolean mShowRender = false;
    private boolean mCameraChanging = false;
    private boolean mIsChangingPreviewSize = false;
    private boolean mIsCreateDetectorSucceeded = false;

    private ExecutorService mExecutor;
    private AtomicBoolean mOnSubjectUpdate = new AtomicBoolean(false);

    private Paint mPaint;
    private Handler mProcessHandler;
    private SurfaceView mSurfaceViewOverlap;


    /*
     * tick counter
     */
    private static int mFps;
    private int mCount = 0;
    private int mFrameCost = 0;
    private long mCurrentTime = 0;
    private boolean mIsFirstCount = true;


    /*
     * Face Detector
     */
    private UnifiedMap mFaceInfo;          // for UI
    private INDArray mFacePoints;
    private INDArray mIrisPoints;
    private INDArray mFaceDetails;
    private GazeActivity mUIActivity;
    private SubjectProxy mSubjectProxy;
    private long mDetectorFlags;
    private Object mTrackingLock = new Object();
    private STMobileHumanActionNative mDetectorNative;


    /*
     * init
     */
    public GazeDisplay(Context context, GLSurfaceView glSurfaceView, SurfaceView surfaceView) {
        initTrackerComponent();
        initFaceDetector();

        mContext = context;
        mCameraProxy = new CameraProxy(context);

        setRenderGlView(glSurfaceView);

        mSurfaceViewOverlap = surfaceView;
        mSurfaceViewOverlap.setZOrderMediaOverlay(true);
        mSurfaceViewOverlap.getHolder().setFormat(PixelFormat.TRANSLUCENT);

        initPainter();
        mGLRender = new GLRender();

        runFaceDetector(getFaceDetectorModelFlags());
        setCalibrationParam();
    }

    private void setCalibrationParam() {
        double[] params = new double[]{
            // -10.5, 5.0, 30, 0, 0
            10.0, 6.50, 29.50, -6.0, 0
        };

        mSubjectProxy.setCalibrationParams(params);
    }

    private void initTrackerComponent() {
        mSubjectProxy = new SubjectProxy();
        mFaceInfo = UnifiedMap.newMap();
        mFacePoints = toZeroMatrix(2, 106);
        mIrisPoints = toZeroMatrix(2, 38);
        mFaceDetails = toZeroMatrix(2, 134);
        mExecutor = Executors.newFixedThreadPool(2);
    }

    private void setRenderGlView(GLSurfaceView glView) {
        mGlSurfaceView = glView;
        mGlSurfaceView.setEGLContextClientVersion(2);
        mGlSurfaceView.setRenderer(this);
        mGlSurfaceView.setRenderMode(GLSurfaceView.RENDERMODE_WHEN_DIRTY);
    }

    private void initFaceDetector() {
        mDetectorFlags =
            STMobileHumanActionNative.ST_MOBILE_FACE_DETECT |
            STMobileHumanActionNative.ST_MOBILE_DETECT_EXTRA_FACE_POINTS|
            STMobileHumanActionNative.ST_MOBILE_DETECT_EYEBALL_CONTOUR;
        mDetectorNative = new STMobileHumanActionNative();
    }

    private void initPainter() {
        int strokeWidth = 10;
        mPaint = new Paint();
        mPaint.setColor(Color.rgb(240, 100, 100));
        mPaint.setStrokeWidth(strokeWidth);
        mPaint.setStyle(Paint.Style.FILL_AND_STROKE);
    }

    private void setUpCamera() {
        if (mTextureId == OpenGLUtils.NO_TEXTURE) {
            mTextureId = OpenGLUtils.getExternalOESTextureID();
            mSurfaceTexture = new SurfaceTexture(mTextureId);
        }

        if (mSupportedPreviewSizes == null || mSupportedPreviewSizes.size() == 0) {
            return;
        }

        String size = mSupportedPreviewSizes.get(mCurrentPreview);
        int index = size.indexOf('x');
        mImageWidth = Integer.parseInt(size.substring(index + 1));
        mImageHeight = Integer.parseInt(size.substring(0, index));
        mFrameSize = mImageHeight * mImageWidth * 3 / 2;
        mCameraProxy.setPreviewSize(mImageHeight, mImageWidth);

        boolean flipHorizontal = mCameraProxy.isFlipHorizontal();
        mGLRender.adjustTextureBuffer(mCameraProxy.getOrientation(), flipHorizontal);

        mCameraProxy.startPreview(mSurfaceTexture, mPreviewCallback);
        mDetectorNative.setParam(FaceDetectorParamsType.CAM_FOVX, mCameraProxy.getAngleOfView());

        mCameraProxy.setCameraFocus(mCameraProxy.getParameters());
    }



    /*
     * Detection Result
     */
    private void setDetectionResult(STHumanAction detectionResult) {
        if (detectionResult == null || detectionResult.faces.length == 0) {
            return;
        }

        STMobileFaceInfo faceInfo = detectionResult.faces[0];
        STPoint[] facePoints = faceInfo.getFace().getPointsArray();
        STPoint[] irisPoints = faceInfo.getEyeballContour();
        STPoint[] faceDetails = faceInfo.getExtraFacePoints();

        if (facePoints.length == 0 || irisPoints.length == 0) {
            return;
        }

        if (mOnSubjectUpdate.get()) {
            return;
        } else {
            mOnSubjectUpdate.set(true);
            updateSubject(facePoints, irisPoints, faceDetails);
        }
    }

    private void updateSubject(STPoint[] facePoints, STPoint[] irisPoints, STPoint[] faceDetails) {
        // Async Mode
        mExecutor.submit(() -> {
            try {
                ZTDetectionParser.intoNDArray(facePoints, mFacePoints);
                ZTDetectionParser.intoNDArray(irisPoints, mIrisPoints);
                ZTDetectionParser.intoNDArray(faceDetails, mFaceDetails);

                mSubjectProxy.updateFace(mFacePoints, mIrisPoints, mFaceDetails);
                updateDebugInfo();
            }
            catch (Exception e) {
                Log.e(TAG, ExceptionUtils.getStackTrace(e));
            }
            finally {
                mOnSubjectUpdate.set(false);
            }
        });

        // Sync Mode
        /*
        mOnSubjectUpdate.set(true);
        try {
            ZTDetectionParser.intoNDArray(facePoints, mFacePoints);
            ZTDetectionParser.intoNDArray(irisPoints, mIrisPoints);
            ZTDetectionParser.intoNDArray(faceDetails, mFaceDetails);

            mSubjectProxy.updateFace(mFacePoints, mIrisPoints, mFaceDetails);
            updateDebugInfo();
        } catch (Exception e) {
            Log.e(TAG, ExceptionUtils.getStackTrace(e));
        } finally {
            mOnSubjectUpdate.set(false);
        }
        */
    }


    /*
     * Debug Info
     */
    double rx, ry, rz;
    double lx, ly, lz;
    double tx, ty, tz;
    double rix, riy, riz;
    double lix, liy, liz;

    private void updateDebugInfo() {
        INDArray ER = mSubjectProxy.getCenterOfEyeRotation(true);
        INDArray EL = mSubjectProxy.getCenterOfEyeRotation(false);

        if (ER != null) {
            rx = ER.getDouble(0, 0);
            ry = ER.getDouble(1, 0);
            rz = ER.getDouble(2, 0);
            lx = EL.getDouble(0, 0);
            ly = EL.getDouble(1, 0);
            lz = EL.getDouble(2, 0);

            mUIActivity.drawTarget(rx, ry); // R이 나의 왼쪽 눈
            mUIActivity.drawThirdTarget(lx, ly);

            // mFaceInfo.put("ER", String.format("%1.1f %1.1f %1.1f", rx, ry, rz));
            // mFaceInfo.put("ER", String.format("%1.1f %1.1f %1.1f\n%1.1f %1.1f %1.1f", rx, ry, rz, lx, ly, lz));
        }

        INDArray IR = mSubjectProxy.getIrisCenterOfRightEye(true);
        INDArray IL = mSubjectProxy.getIrisCenterOfRightEye(false);

        if (IR != null) {
            rix = IR.getDouble(0, 0);
            riy = IR.getDouble(1, 0);
            riz = IR.getDouble(2, 0);
            // lix = IL.getDouble(0, 0);
            // liy = IL.getDouble(1, 0);
            // liz = IL.getDouble(2, 0);
            // mFaceInfo.put("IR", String.format("%1.1f %1.1f %1.1f", rix, riy, rz));
        }
        // mFaceInfo.put("IR", String.format("%1.1f %1.1f %1.1f\n%1.1f %1.1f %1.1f", rix, riy, riz, lix, liy, liz));

        INDArray CCR = mSubjectProxy.getCenterOfCornea();
        if (CCR != null) {
            double ccrx = CCR.getDouble(0, 0);
            double ccry = CCR.getDouble(1, 0);
            // mUIActivity.drawTarget(ccrx, ccry);
        }

        ExtrinsicParam EP = mSubjectProxy.getExtrinsicParam();
        INDArray T = EP.getTranslationVec();
        tx = T.getDouble(0, 0);
        ty = T.getDouble(1, 0);
        tz = T.getDouble(2, 0);

        if (T != null) {
            mFaceInfo.put("EP", String.format("%1.1f %1.1f %1.1f", tx, ty, tz));
        }

        double[] mXY = mSubjectProxy.getPointOfGazeOfRightEye();
        mFaceInfo.put("PR", String.format("%1.1f, %1.1f", mXY[0], mXY[1]));

        // mFaceInfo.put("YPR", String.format("%1.1f, %1.1f, %1.1f", mYaw, mPitch, mRoll));

        mUIActivity.drawSecondTarget(mXY[0], mXY[1]);
        mUIActivity.endDraw();
    }


    /*
     * Face Detector
     */
    private int getFaceDetectorModelFlags() {
        return STCommonNative.ST_MOBILE_TRACKING_MULTI_THREAD
            | STCommonNative.ST_MOBILE_TRACKING_ENABLE_DEBOUNCE
            | STMobileHumanActionNative.ST_MOBILE_ENABLE_FACE_DETECT
            | STMobileHumanActionNative.ST_MOBILE_DETECT_MODE_VIDEO;
    }

    private void runFaceDetector(int modelFlags) {
        String faceBase = FileUtils.FACE_TRACK_MODEL_NAME;
        String faceDetail = FileUtils.FACE_DETAIL_MODEL_NAME;
        String eyeContour = FileUtils.EYEBALL_CONTOUR_MODEL_NAME;

        new Thread(() -> {
            synchronized (mTrackingLock) {
                int result = mDetectorNative.createInstanceFromAssetFile(faceBase, modelFlags, mContext.getAssets());
                result += mDetectorNative.addSubModelFromAssetFile(faceDetail, mContext.getAssets());
                result += mDetectorNative.addSubModelFromAssetFile(eyeContour, mContext.getAssets());

                mDetectorNative.addSubModelFromAssetFile(FileUtils.EYEBALL_CONTOUR_MODEL_NAME, mContext.getAssets());

                LogUtils.i(TAG, "*** Create detector result: %d ***", result);

                if (result == 0) {
                    mIsCreateDetectorSucceeded = true;
                    mDetectorNative.setParam(FaceDetectorParamsType.BACKGROUND_RESULT_ROTATE, 1.0f);
                }
            }
        }).start();

        HandlerThread mHandlerThread = new HandlerThread("ProcessImageThread");
        mHandlerThread.start();
        mProcessHandler = new Handler(mHandlerThread.getLooper()) {
            @Override
            public void handleMessage(Message msg) {
                if (msg.what == MESSAGE_PROCESS_IMAGE && !mIsPaused && mIsCreateDetectorSucceeded) {
                    processDetection();
                    mGlSurfaceView.requestRender();
                }
            }
        };
    }

    private Camera.PreviewCallback mPreviewCallback = new Camera.PreviewCallback() {
        public void onPreviewFrame(final byte[] data, Camera camera) {
            if (mCameraChanging || mCameraProxy.getCamera() == null) {
                return;
            }

            if (mImageData == null || mImageData.length != mFrameSize) {
                mImageData = new byte[mFrameSize];
            }

            synchronized (mImageDataLock) {
                System.arraycopy(data, 0, mImageData, 0, data.length);
            }

            mProcessHandler.removeMessages(MESSAGE_PROCESS_IMAGE);
            mProcessHandler.sendEmptyMessage(MESSAGE_PROCESS_IMAGE);
        }
    };

    private void processDetection() {
        if (mTmpBuffer == null || mTmpBuffer.length != mFrameSize) {
            mTmpBuffer = new byte[mFrameSize];
        }

        if (mCameraChanging || mTmpBuffer.length != mImageData.length) {
            clearOverLap();
            return;
        }

        synchronized (mImageDataLock) {
            System.arraycopy(mImageData, 0, mTmpBuffer, 0, mImageData.length);
        }

        int dir = Accelerometer.getDirection();
        if (((mCameraProxy.getOrientation() == 270 && (dir & 1) == 1) ||
             (mCameraProxy.getOrientation() == 90 && (dir & 1) == 0))) {
            dir = (dir ^ 2);
        }

        long startTime = System.currentTimeMillis();
        STHumanAction detectResult = mDetectorNative.humanActionDetect(mTmpBuffer,
                CAPTURE_FORMAT, mDetectorFlags, dir, mImageHeight, mImageWidth);
        LogUtils.i(TAG, "detector cost time: %d", System.currentTimeMillis() - startTime);

        if (mCameraChanging || mIsPaused) {
            setDetectionResult(null);
            clearOverLap();
            return;
        }

        detectResult = postProcessResult(detectResult, mCameraProxy.getOrientation());
        mFrameCost = (int) (System.currentTimeMillis() - startTime);
        computeFps();

        setDetectionResult(detectResult);

        if (detectResult != null) {
            drawDetectionResults(detectResult, mImageWidth, mImageHeight);
        }
    }



    /*
     * Model result
     */
    private STHumanAction postProcessResult(STHumanAction detection, int cameraOrientation) {
        if (detection == null) {
            return null;
        }

        int imgWidth = mImageWidth;
        int imgHeight = mImageHeight;
        int rotate = STRotateType.ST_CLOCKWISE_ROTATE_90;

        if (cameraOrientation == 270) {
            rotate =  STRotateType.ST_CLOCKWISE_ROTATE_270;
        }

        detection = STHumanAction.humanActionRotate(imgHeight, imgWidth, rotate, true, detection);
        detection = STHumanAction.humanActionMirror(imgWidth, detection);

        return detection;
    }

    private void drawDetectionResults(STHumanAction detectionResult, int width, int height) {
        if (!mSurfaceViewOverlap.getHolder().getSurface().isValid() || mCameraChanging) {
            return;
        }

        Canvas canvas = mSurfaceViewOverlap.getHolder().lockCanvas();

        if (canvas == null) {
            return;
        }

        canvas.drawColor(0, PorterDuff.Mode.CLEAR);

        if (!mShowRender || detectionResult.faceCount <= 0) {
            mSurfaceViewOverlap.getHolder().unlockCanvasAndPost(canvas);
            return;
        }

        for (STMobileFaceInfo face : detectionResult.faces) {
            drawFace(face, width, height, canvas);
        }

        mSurfaceViewOverlap.getHolder().unlockCanvasAndPost(canvas);
    }

    private void drawFace(STMobileFaceInfo face, int width, int height, Canvas canvas) {
        STPoint[] face106Points = face.getFace().getPointsArray();
        PointF[] rotatedFace106Points = new PointF[face106Points.length];
        for (int i = 0; i < face106Points.length; i++) {
            rotatedFace106Points[i] = new PointF(face106Points[i].getX(), face106Points[i].getY());
        }

        float[] visibles = face.getFace().getVisibilityArray();
        Rect faceRect = face.getFace().getRect().convertToRect();

        if (mCameraChanging) {
            return;
        }

        DetectorUtils.drawFaceKeyPoints(canvas, mPaint, rotatedFace106Points, visibles, width, height, Color.parseColor("#00ee00"));
        DetectorUtils.drawFaceRect(canvas, faceRect, width, height);

        //extra face info
        if (face.extraFacePointsCount > 0) {
            STPoint[] extraFacePoints = face.getExtraFacePoints();
            PointF[] rotatedExtraFacePoints = new PointF[extraFacePoints.length];
            for (int i = 0; i < extraFacePoints.length; i++) {
                rotatedExtraFacePoints[i] = new PointF(extraFacePoints[i].getX(), extraFacePoints[i].getY());
            }

            DetectorUtils.drawFaceKeyPoints(canvas, mPaint, rotatedExtraFacePoints, null, width, height, Color.parseColor("#0a8dff"));
        }

        //eyeball center
        if (face.eyeballCenterPointsCount == 2) {
            STPoint[] eyeballCenterPoints = face.getEyeballCenter();
            PointF[] leftEyeballCenterPoints = new PointF[1];
            PointF[] rightEyeballCenterPoints = new PointF[1];
            leftEyeballCenterPoints[0] = new PointF(eyeballCenterPoints[0].getX(), eyeballCenterPoints[0].getY());
            rightEyeballCenterPoints[0] = new PointF(eyeballCenterPoints[1].getX(), eyeballCenterPoints[1].getY());

            float value = 0.8f;
            if (face.leftEyeballScore >= value) {
                DetectorUtils.drawFaceKeyPoints(canvas, mPaint, leftEyeballCenterPoints, null, width, height, Color.parseColor("#ff00f6"));
            }

            if (face.rightEyeballScore >= value) {
                DetectorUtils.drawFaceKeyPoints(canvas, mPaint, rightEyeballCenterPoints, null, width, height, Color.parseColor("#ff00f6"));
            }
        }

        //eyeball contour
        if (face.eyeballContourPointsCount == 38) {
            STPoint[] eyeballContourPoints = face.getEyeballContour();
            PointF[] leftEyeballContourPoints = new PointF[eyeballContourPoints.length / 2];
            PointF[] rightEyeballContourPoints = new PointF[eyeballContourPoints.length / 2];

            for (int i = 0; i < eyeballContourPoints.length / 2; i++) {
                leftEyeballContourPoints[i] = new PointF(eyeballContourPoints[i].getX(), eyeballContourPoints[i].getY());
            }

            for (int i = eyeballContourPoints.length / 2; i < face.eyeballContourPointsCount; i++) {
                rightEyeballContourPoints[i - eyeballContourPoints.length / 2] = new PointF(eyeballContourPoints[i].getX(), eyeballContourPoints[i].getY());
            }

            float value = 0.8f;
            if (face.leftEyeballScore >= value) {
                DetectorUtils.drawFaceKeyPoints(canvas, mPaint, leftEyeballContourPoints, null, width, height, Color.parseColor("#ffe763"));
            }

            if (face.rightEyeballScore >= value) {
                DetectorUtils.drawFaceKeyPoints(canvas, mPaint, rightEyeballContourPoints, null, width, height, Color.parseColor("#ffe763"));
            }
        }
    }

    private void clearOverLap() {
        if (!mSurfaceViewOverlap.getHolder().getSurface().isValid()) {
            return;
        }

        Canvas canvas = mSurfaceViewOverlap.getHolder().lockCanvas();

        if (canvas == null) {
            return;
        }

        canvas.drawColor(0, PorterDuff.Mode.CLEAR);
        mSurfaceViewOverlap.getHolder().unlockCanvasAndPost(canvas);
    }


    /*
     * miscellaneous
     */
    public void setUI(GazeActivity activity) {
        mUIActivity = activity;
    }

    private void computeFps() {
        long timer = System.currentTimeMillis();
        mCount++;
        if (mIsFirstCount) {
            mCurrentTime = timer;
            mIsFirstCount = false;
        } else {
            int cost = (int) (timer - mCurrentTime);
            if (cost >= 1000) {
                mCurrentTime = timer;
                mFps = (mCount * 1000) / cost;
                mCount = 0;
            }
        }
        LogUtils.i(TAG, "fps: %d", mFps);
    }

    public int getFps() {
        return mFps;
    }

    public UnifiedMap getFaceInfo() {
        return mFaceInfo;
    }

    public boolean isChangingPreviewSize() {
        return mIsChangingPreviewSize;
    }


    /*
     * Life Cycle
     */
    public void onResume() {
        LogUtils.i(TAG, "onResume");
        mIsPaused = false;
        if (mCameraProxy.getCamera() == null) {
            if (mCameraProxy.getNumberOfCameras() == 1) {
                mCameraID = Camera.CameraInfo.CAMERA_FACING_BACK;
            }
            mCameraProxy.openCamera(mCameraID);
            mSupportedPreviewSizes = mCameraProxy.getSupportedPreviewSize(new String[]{"1280x720", "640x480"});
        }

        mGLRender = new GLRender();

        mGlSurfaceView.onResume();
        mGlSurfaceView.forceLayout();
        mGlSurfaceView.requestRender();
    }

    public void onPause() {
        LogUtils.i(TAG, "onPause");
        mIsPaused = true;
        mCameraProxy.releaseCamera();

        mGlSurfaceView.queueEvent(() -> {
            mDetectorNative.reset();
            deleteTextures();
            if (mSurfaceTexture != null) {
                mSurfaceTexture.release();
            }
            mGLRender.destroyFrameBuffers();
        });

        mGlSurfaceView.onPause();
    }

    public void onDestroy() {
        mImageData = null;
        mTmpBuffer = null;
        synchronized (mTrackingLock) {
            mDetectorNative.destroyInstance();
        }
    }

    private void deleteTextures() {
        if (mTextureId != OpenGLUtils.NO_TEXTURE) {
            GLES20.glDeleteTextures(1, new int[]{mTextureId}, 0);
        }

        mTextureId = OpenGLUtils.NO_TEXTURE;
        mGLRender.deleteSegmentResultTexture();
    }


    /*
     * Render
     */
    public void changePreviewSize(int currentPreview) {
        if (mCameraProxy.getCamera() == null || mCameraChanging || mIsPaused) {
            return;
        }

        mCurrentPreview = currentPreview;
        mIsChangingPreviewSize = true;
        mCameraChanging = true;

        mCameraProxy.stopPreview();
        mGlSurfaceView.queueEvent(() -> {
            mDetectorNative.reset();

            deleteTextures();
            if (mCameraProxy.getCamera() != null) {
                setUpCamera();
            }

            mGLRender.destroySegmentFrameBuffer();
            mGLRender.init(mImageWidth, mImageHeight);
            mGLRender.initDrawBackGround();
            mGLRender.calculateVertexBuffer(mSurfaceWidth, mSurfaceHeight, mImageWidth, mImageHeight);

            mCameraChanging = false;
            mIsChangingPreviewSize = false;
            LogUtils.d(TAG, "exit  change Preview size queue event");
        });
    }

    private void adjustViewPort(int width, int height) {
        mSurfaceHeight = height;
        mSurfaceWidth = width;
        GLES20.glViewport(0, 0, mSurfaceWidth, mSurfaceHeight);
        mGLRender.calculateVertexBuffer(mSurfaceWidth, mSurfaceHeight, mImageWidth, mImageHeight);
    }

    @Override
    public void onSurfaceCreated(GL10 gl, EGLConfig config) {
        LogUtils.i(TAG, "onSurfaceCreated");
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
        LogUtils.i(TAG, "onSurfaceChanged");
        if (mIsPaused) {
            return;
        }
        adjustViewPort(width, height);
        mGLRender.init(mImageWidth, mImageHeight);
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


    // View Toggle
    public void toggleRender() {
        mShowRender = !mShowRender;
    }

    public void setShowRender(boolean enable) {
        mShowRender = enable;
    }

}
