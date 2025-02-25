package camp.visual.kappa.display;

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
import android.opengl.GLSurfaceView.Renderer;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Message;
import android.util.Log;
import android.util.Pair;
import android.view.SurfaceView;

import com.sensetime.stmobile.FaceDetectorParamsType;
import com.sensetime.stmobile.STCommonNative;
import com.sensetime.stmobile.STMobileHumanActionNative;
import com.sensetime.stmobile.STRotateType;
import com.sensetime.stmobile.model.STHumanAction;
import com.sensetime.stmobile.model.STMobileFaceInfo;
import com.sensetime.stmobile.model.STPoint;

import org.apache.commons.lang3.exception.ExceptionUtils;
import org.eclipse.collections.impl.list.mutable.FastList;
import org.eclipse.collections.impl.map.mutable.UnifiedMap;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicBoolean;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

import camp.visual.device.Devices;
import camp.visual.kappa.CameraActivity;
import camp.visual.kappa.al.SubjectProxy;
import camp.visual.kappa.cali.CalibrationScenario;
import camp.visual.kappa.cali.Calibrator;
import camp.visual.kappa.cali.Snapshot;
import camp.visual.camera.CameraProxy;
import camp.visual.camera.ExtrinsicParam;
import camp.visual.kappa.glutils.OpenGLUtils;
import camp.visual.kappa.utils.Accelerometer;
import camp.visual.kappa.utils.DetectorUtils;
import camp.visual.kappa.utils.FileUtils;
import camp.visual.kappa.utils.LogUtils;
import camp.visual.kappa.zt.ZTDetectionParser;

import static camp.visual.ac.Maths.toZeroMatrix;
import static org.apache.commons.math3.util.Precision.round;


public class CameraDisplay implements Renderer {
    private String TAG = "CameraDisplay";

    private static final int MESSAGE_PROCESS_IMAGE = 100;
    private static final int MESSAGE_ADD_SUB_MODEL = 1001;
    private static final int MESSAGE_REMOVE_SUB_MODEL = 1002;

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
    private boolean mIsSnapshotMode = false;                // Snapshot Mode
    private boolean mIsChangingPreviewSize = false;

    private ExecutorService mExecutor;
    private AtomicBoolean mOnCalibration = new AtomicBoolean(true);
    private AtomicBoolean mOnSubjectUpdate = new AtomicBoolean(false);
    private AtomicBoolean mSnapshotRequested = new AtomicBoolean(false);
    public AtomicBoolean mCalibrationSuccess = new AtomicBoolean(false);

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
     * Subject Tracker & Calibrator
     */
    private Calibrator mCalibrator;
    private Future mCalibrationTask;
    private CameraActivity mUIActivity;
    private SubjectProxy mSubjectProxy;
    private FastList<Snapshot> mSnapshots;
    private CalibrationScenario mCaliScenario;


    /*
     * Detection result Holder
     */
    private double mYaw;
    private double mPitch;
    private double mRoll;
    private UnifiedMap mFaceInfo;          // for UI
    private INDArray mFacePoints;
    private INDArray mIrisPoints;
    private INDArray mFaceDetails;


    /*
     * Face Detector
     */
    private long mDetectorFlags;
    private final Object mTrackingLock = new Object();
    private Handler mSubModelsManagerHandler;
    private STMobileHumanActionNative mDetectorNative;


    /*
     * init
     */
    public CameraDisplay(Context context, GLSurfaceView glSurfaceView, SurfaceView surfaceView) {
        initTrackerComponent();
        initFaceDetector();
        initCalibrator();

        mContext = context;
        mCameraProxy = new CameraProxy(context);

        setRenderGlView(glSurfaceView);

        mSurfaceViewOverlap = surfaceView;
        mSurfaceViewOverlap.setZOrderMediaOverlay(true);
        mSurfaceViewOverlap.getHolder().setFormat(PixelFormat.TRANSLUCENT);

        initPainter();
        mGLRender = new GLRender();

        runFaceDetector(getFaceDetectorModelFlags());

        // For Future Option
        // initSubModelManager();

        // *** For Debug ***
        // setCalibrationParam();
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
        mDetectorFlags = STMobileHumanActionNative.ST_MOBILE_FACE_DETECT
                | STMobileHumanActionNative.ST_MOBILE_DETECT_EYEBALL_CONTOUR
                | STMobileHumanActionNative.ST_MOBILE_DETECT_EXTRA_FACE_POINTS;
        mDetectorNative = new STMobileHumanActionNative();
    }

    private void initPainter() {
        int strokeWidth = 10;
        mPaint = new Paint();
        mPaint.setColor(Color.rgb(240, 100, 100));
        mPaint.setStrokeWidth(strokeWidth);
        mPaint.setStyle(Paint.Style.FILL_AND_STROKE);
    }

    private void initSubModelManager() {
        HandlerThread mSubModelsManagerThread = new HandlerThread("SMT");
        mSubModelsManagerThread.start();
        mSubModelsManagerHandler = new Handler(mSubModelsManagerThread.getLooper()) {
            public void handleMessage(Message msg) {
                if (mIsPaused || mCameraChanging) {
                    return;
                }

                if (msg.what == MESSAGE_ADD_SUB_MODEL && msg.obj != null) {
                    addSubModel((String) msg.obj);
                } else if (msg.what == MESSAGE_REMOVE_SUB_MODEL && (int) msg.obj != 0) {
                    removeSubModel((int) msg.obj);
                }
            }
        };
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

    private void initCalibrator() {
        mSnapshots = FastList.newList();
        mCalibrator = new Calibrator(new ZTDetectionParser());
        mCaliScenario = new CalibrationScenario();
    }


    /*
     * Detection Result
     */
    private void setDetectionResult(STHumanAction detectionResult) {
        if (mIsSnapshotMode) {
            Snapshot s = mSnapshots.getLast();
            updateSubject(s.getFacePoints(), s.getIrisPoints(), s.getFaceDetails());
            return;
        }

        if (detectionResult == null || detectionResult.faces.length == 0) {
            return;
        }

        STMobileFaceInfo faceInfo = detectionResult.faces[0];
        STPoint[] facePoints = faceInfo.getFace().getPointsArray();
        STPoint[] irisPoints = faceInfo.getEyeballContour();
        STPoint[] faceDetails = faceInfo.getExtraFacePoints();

        mYaw = (Math.round(faceInfo.face106.getYaw() * 100)) / 100;
        mPitch = (Math.round(faceInfo.face106.getPitch() * 100)) / 100;
        mRoll = (Math.round(faceInfo.face106.getRoll() * 100)) / 100;

        if (facePoints.length == 0 || irisPoints.length == 0) {
            return;
        }

        if (mOnSubjectUpdate.get()) {
            return;
        }

        updateSubject(facePoints, irisPoints, faceDetails);
    }

    private void updateSubject(STPoint[] facePoints, STPoint[] irisPoints, STPoint[] faceDetails) {
        try {
            mOnSubjectUpdate.set(true);
            ZTDetectionParser.intoNDArray(facePoints, mFacePoints);
            ZTDetectionParser.intoNDArray(irisPoints, mIrisPoints);
            ZTDetectionParser.intoNDArray(faceDetails, mFaceDetails);

            // 시선 추적 과정
            mSubjectProxy.updateFace(mFacePoints, mIrisPoints, mFaceDetails);
            updateDebugInfo();

            // 얼굴 정위 체크
            // mSubjectProxy.checkFacePose(mFacePoints, mFaceDetails);

        }
        catch (Exception e) {
            Log.e(TAG, ExceptionUtils.getStackTrace(e));
        }
        finally {
            mOnSubjectUpdate.set(false);
        }
    }

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
        INDArray R = EP.getRotationMat();
        tx = T.getDouble(0, 0);
        ty = T.getDouble(1, 0);
        tz = T.getDouble(2, 0);

        if (T != null) {
            mFaceInfo.put("EP", String.format("%1.1f %1.1f %1.1f", tx, ty, tz));
        }

        double[] mXY = mSubjectProxy.getPointOfGazeOfRightEye();
        mFaceInfo.put("PR", String.format("%1.1f, %1.1f", mXY[0], mXY[1]));
        // mFaceInfo.put("PR", toStringMat3x3(R));

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

                LogUtils.i(TAG, "*** Detector result: %d ***", result);

                if (result == 0) {
                    mDetectorNative.setParam(FaceDetectorParamsType.BACKGROUND_RESULT_ROTATE, 1.0f);
                }
            }
        }).start();

        HandlerThread mHandlerThread = new HandlerThread("ProcessImageThread");
        mHandlerThread.start();
        mProcessHandler = new Handler(mHandlerThread.getLooper()) {
            @Override
            public void handleMessage(Message msg) {
                if (msg.what == MESSAGE_PROCESS_IMAGE && !mIsPaused) {
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

        if (mOnCalibration.get()) {
            handleSnapshot(detectResult);
        } else if (mCalibrationSuccess.get()) {
            setDetectionResult(detectResult);
        }

        if (detectResult != null) {
            drawDetectionResults(detectResult, mImageWidth, mImageHeight);
        }
    }


    /*
     * Calibration & Snapshot
     */
    private void handleSnapshot(STHumanAction detection) {
        if (!mSnapshotRequested.get()) {
            return;
        }

        int idx = mSnapshots.size();
        Log.e(TAG, String.format("*** take takeSnapshot %d ***", idx));

        Snapshot s = new Snapshot(mFrameSize, idx);
        s.setCameraParams(Devices.getDevice().getCameraParamDup());
        Pair<Double, Double> mxy = mCaliScenario.getCurrentTarget(idx);
        s.setCamXY(mxy.first, mxy.second);

        if (detection.faceCount <= 0) {
            s.setFaceDetectFail();
            Log.e(TAG, String.format("*** face detect failed %d ***", s.mIdx));
        }

        STMobileFaceInfo faceInfo = detection.faces[0];
        STPoint[] facePoints = faceInfo.getFace().getPointsArray();
        STPoint[] irisPoints = faceInfo.getEyeballContour();
        STPoint[] faceDetails = faceInfo.getExtraFacePoints();

        s.updateFacePoints(facePoints, irisPoints, faceDetails);

        mSnapshots.add(s);
        mSnapshotRequested.set(false);
        mUIActivity.updateUI();
    }

    public void requestSnapshot() {
        mSnapshotRequested.set(true);
    }

    public void requestCalibration() {
        Log.e(TAG, "*** process snapshot ***");

        for (Snapshot s : mSnapshots) {
            if (s.mIsDetectionSuccess) {
                mCalibrator.add(s);
            }
        }

        mCalibrationTask = mExecutor.submit(() -> {
            Log.e(TAG, "*** run calibration ***");

            try {
                mCalibrator.preprocess();
                mSubjectProxy.setCalibrationParams(mCalibrator.calibrate());
                mCalibrationSuccess.set(true);
            } catch (Exception e) {
                Log.e(TAG, ExceptionUtils.getStackTrace(e));
            } finally {
                mOnCalibration.set(false);
            }
        });
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
        if (canvas == null)
            return;

        canvas.drawColor(0, PorterDuff.Mode.CLEAR);
        mSurfaceViewOverlap.getHolder().unlockCanvasAndPost(canvas);
    }


    /*
     * miscellaneous
     */
    public void setUI(CameraActivity activity) {
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

    public int getFrameCost() {
        return mFrameCost;
    }

    public UnifiedMap getFaceInfo() {
        return mFaceInfo;
    }

    public boolean isChangingPreviewSize() {
        return mIsChangingPreviewSize;
    }

    public int getNumSnapshot() {
        return mSnapshots.size();
    }

    public int getNumScenario() {
       return mCaliScenario.getNumScenario();
    }

    public Pair<Double, Double> getCurrentTarget() {
        return mCaliScenario.getCurrentTarget(mSnapshots.size());
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

        if (mCalibrationTask != null) {
            mCalibrationTask.cancel(true);
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



    /*
     * model switch
     */
    public void toggleRender() {
        mShowRender = !mShowRender;
    }

    public void setShowRender(boolean enable) {
        mShowRender = enable;
    }

    public void setEnableFace106(boolean enable) {
        if (enable) {
            mDetectorFlags |= STMobileHumanActionNative.ST_MOBILE_FACE_DETECT;
        } else {
            mDetectorFlags &= ~STMobileHumanActionNative.ST_MOBILE_FACE_DETECT;
        }
    }

    public void setEnableFaceExtra(boolean enable) {
        if (enable) {
            mDetectorFlags |= STMobileHumanActionNative.ST_MOBILE_DETECT_EXTRA_FACE_POINTS;

            Message msg = mSubModelsManagerHandler.obtainMessage(MESSAGE_ADD_SUB_MODEL);
            msg.obj = FileUtils.FACE_DETAIL_MODEL_NAME;
            mSubModelsManagerHandler.sendMessage(msg);
        } else {
            mDetectorFlags &= ~STMobileHumanActionNative.ST_MOBILE_DETECT_EXTRA_FACE_POINTS;
        }
    }

    public void setEnableEyeBallCenter(boolean enable) {
        if (enable) {
            mDetectorFlags |= STMobileHumanActionNative.ST_MOBILE_DETECT_EYEBALL_CENTER;

            Message msg = mSubModelsManagerHandler.obtainMessage(MESSAGE_ADD_SUB_MODEL);
            msg.obj = FileUtils.EYEBALL_CONTOUR_MODEL_NAME;
            mSubModelsManagerHandler.sendMessage(msg);
        } else {
            mDetectorFlags &= ~STMobileHumanActionNative.ST_MOBILE_DETECT_EYEBALL_CENTER;
        }
    }

    public void setEnableEyeBallContour(boolean enable) {
        if (enable) {
            mDetectorFlags |= STMobileHumanActionNative.ST_MOBILE_DETECT_EYEBALL_CONTOUR;

            Message msg = mSubModelsManagerHandler.obtainMessage(MESSAGE_ADD_SUB_MODEL);
            msg.obj = FileUtils.EYEBALL_CONTOUR_MODEL_NAME;
            mSubModelsManagerHandler.sendMessage(msg);
        } else {
            mDetectorFlags &= ~STMobileHumanActionNative.ST_MOBILE_DETECT_EYEBALL_CONTOUR;
        }
    }



    /*
     * Model
     */
    private void addSubModel(final String modelName) {
        synchronized (mTrackingLock) {
            int result = mDetectorNative.addSubModelFromAssetFile(modelName, mContext.getAssets());
            LogUtils.i(TAG, "add sub model result: %d", result);
        }
    }

    private void removeSubModel(final int config) {
        synchronized (mTrackingLock) {
            int result = mDetectorNative.removeSubModelByConfig(config);
            LogUtils.i(TAG, "remove sub model result: %d", result);

            if (config == STMobileHumanActionNative.ST_MOBILE_ENABLE_FACE_EXTRA_DETECT) {
                mDetectorFlags &= ~STMobileHumanActionNative.ST_MOBILE_DETECT_EXTRA_FACE_POINTS;
            }
        }
    }

}