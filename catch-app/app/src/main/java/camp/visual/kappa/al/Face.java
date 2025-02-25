package camp.visual.kappa.al;

import android.util.Log;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfPoint3f;
import org.opencv.core.Point;
import org.opencv.core.Point3;

import camp.visual.camera.CameraParam;
import camp.visual.camera.ExtrinsicParam;
import camp.visual.kappa.ds.face.ZTPerspectiveMapper;

import static camp.visual.ac.Maths.identityMatrix;
import static camp.visual.ac.Maths.muliToElementIn;
import static camp.visual.ac.Maths.toStringMat3x3;
import static camp.visual.ac.Maths.toZeroMatrix;
import static camp.visual.ac.cv2.CvHelper.toExtrinsicMat;
import static org.opencv.calib3d.Calib3d.CV_EPNP;
import static org.opencv.calib3d.Calib3d.solvePnP;

public class Face {
    private String TAG = "FACE";


    /**
     * NOTE: DO NOT RELEASE WHILE APP RUNNING
     */
    private Mat mRotVec = new Mat();
    private Mat mTransVec = new Mat();
    private ZTPerspectiveMapper mMapper;                // TODO: Interface based


    /**
     * Status
     */
    private EyeBall mLeftEyeBall;
    private EyeBall mRightEyeBall;
    private CameraParam mCameraParams;
    private INDArray mFaceCoordinateSystem;             //
    private INDArray mLensToIrisCenterUnitVec;          // [R[3], L[3]]
    private INDArray mEyeCenterOfRotationInCamera;      // [Col_R[3], Col_L[3]] E


    /**
     * Native Cache Var
     */
    private static MatOfPoint2f mImgMat = new MatOfPoint2f();
    private static MatOfPoint3f mObjMat = new MatOfPoint3f();


    /**
     * Profile Unique Param
     */
    private INDArray mEyeRLCenterOfRotationCorrection;  // [R[3], L[3]] (Anchor와의 변위차, 가운데 중심 대칭)

    public Face(CameraParam cp, ZTPerspectiveMapper faceMapper) {
        mCameraParams = cp;
        mMapper = faceMapper;
        mLeftEyeBall = new EyeBall(false);
        mRightEyeBall = new EyeBall(true);
        mFaceCoordinateSystem = identityMatrix(3);
        setEyeCenterOfRotationCorrection(toZeroMatrix(3, 2));
    }



    /**
     * APIs
     */
    public void updateFace(INDArray facePoints, INDArray irisCenters, INDArray faceDetails) {
        Point[] imagePoints = mMapper.imagePointsFrom(facePoints, faceDetails);
        Point3[] modelPoints = mMapper.get3dModelPoints();
        INDArray caruncles = mMapper.carunclesFrom(faceDetails);

        try {
            boolean res = computeFacePosition(imagePoints, modelPoints, mCameraParams, mRotVec, mTransVec);

            if (res) {
                mCameraParams.setExtrinsicParam(toExtrinsicMat(mRotVec, mTransVec));
                this.propagateUpdate(irisCenters, caruncles);
            } else {
                Log.e(TAG, "No Result In SolvePnP");
            }
        }
        catch (Exception e) {
            Log.e(TAG, e.getMessage());
        }
    }

    private void propagateUpdate(INDArray irisCenterPoints, INDArray carunclesPoints) {
        mFaceCoordinateSystem = computeRotationMatFCSOnCCS(mCameraParams, mFaceCoordinateSystem);
        mLensToIrisCenterUnitVec = computeLensToPixelUVUnitVec(mCameraParams, irisCenterPoints);
        mEyeCenterOfRotationInCamera = computeE(carunclesPoints);

        // mLeftEyeBall.updateReferenceFace(this);
        mRightEyeBall.updateReferenceFace(this);
    }

    public INDArray computeE(INDArray carunclesPoints) {
        INDArray eCorrection = mEyeRLCenterOfRotationCorrection;

        // Version.1 모델의 앵커 위치로 구하는 방법
        // INDArray eyeAnchorPosInWorld = mMapper.getEyeRotationCenterPointsOfModel();
        // INDArray E = computeEyeCenterOfRotationByAnchor(mCameraParams, eyeAnchorPosInWorld, eCorrection);

        // Version.2 Caruncle의 UV와 모델에서 구한 Z를 결합하여 구하는 방법
        INDArray carunclePosInWorld = mMapper.getEyeCarunclesPointsOfModel();
        INDArray E = computeEyeCenterOfRotationByCaruncle(mCameraParams, carunclePosInWorld, carunclesPoints, eCorrection);

        return E;
    }



    /**
     * Algorithms
     */
    private static INDArray computeRotationMatFCSOnCCS(CameraParam cp, INDArray mFCS) {
        // NOTE: Transform Matrix 아닌 단위 방향 벡터만 서술

        // NOTE: Face 단면 평면의 수평, 수직을 구하고 이들의 직교 벡터가 Z축
        // A. PnP를 통해 추출한 rotationMat 에서 변형
        // B. Face pose 여러점 평균하여 계산하는 방법 (o)
        // C. ZT pitch, yaw, roll 그대로 활용하여 계산 (x: 어림 수치인 듯)

        mFCS.assign(cp.getRotationMat());
        return mFCS;
    }

    public static INDArray computeEyeCenterOfRotationByAnchor(CameraParam cp,
                                                              INDArray anchorPosInWorld,
                                                              INDArray profileCorrection) {
        return computeEyeCenterOfRotationByAnchor(cp.getRotationMat(), cp.getTranslationVec(),
            anchorPosInWorld, profileCorrection);
    }

    public static INDArray computeEyeCenterOfRotationByAnchor(INDArray R,
                                                              INDArray T,
                                                              INDArray anchorPosInWorld,
                                                              INDArray profileCorrection) {
        anchorPosInWorld = anchorPosInWorld.add(profileCorrection);
        INDArray anchorInCamera = R.mmul(anchorPosInWorld);
        INDArray eInCamera = anchorInCamera.addColumnVector(T);

        return eInCamera;
    }

    public static INDArray computeEyeCenterOfRotationByCaruncle(CameraParam cp,
                                                                INDArray carunclePosInWorld,
                                                                INDArray carunclePoints,
                                                                INDArray eCorrection) {
        INDArray R = cp.getRotationMat();
        INDArray T = cp.getTranslationVec();
        INDArray inCamera = R.mmul(carunclePosInWorld);
        INDArray posInCamera = inCamera.addColumnVector(T);

        INDArray cuv = computeLensToPixelUVUnitVec(cp, carunclePoints, true);
        double kzr = posInCamera.getDouble(2, 0) / cuv.getDouble(2, 0);
        double kzl = posInCamera.getDouble(2, 1) / cuv.getDouble(2, 1);

        INDArray CUR = cuv.getColumn(0).mul(kzr);
        INDArray CUL = cuv.getColumn(1).mul(kzl);
        INDArray E = Nd4j.hstack(CUR, CUL);

        // Correction
        eCorrection = R.mmul(eCorrection);

        E = E.add(eCorrection);

        return E;
    }

    public static INDArray computeLensToPixelUVUnitVec(CameraParam cp, INDArray pixelPoints, boolean doUndistort) {
        INDArray R = pixelPoints.getColumn(0);
        INDArray L = pixelPoints.getColumn(1);

        INDArray uvUnitVecR;
        INDArray uvUnitVecL;

        if (doUndistort) {
            uvUnitVecR = cp.getUndistortNormalUnitVec(R);
            uvUnitVecL = cp.getUndistortNormalUnitVec(L);
        } else {
            uvUnitVecR = cp.getUVNormalUnitVecFromPixel(R);
            uvUnitVecL = cp.getUVNormalUnitVecFromPixel(L);
        }

        // TODO: Y축 전환을 통합적으로 처리해야 하는데...
        muliToElementIn(uvUnitVecR, 1, 0, -1);
        muliToElementIn(uvUnitVecL, 1, 0, -1);

        return Nd4j.hstack(uvUnitVecR, uvUnitVecL);
    }

    public static INDArray computeLensToPixelUVUnitVec(CameraParam cp, INDArray pixelPoints) {
        return computeLensToPixelUVUnitVec(cp, pixelPoints, true);
    }

    public static boolean computeFacePosition(Point[] imagePoints, Point3[] objectPoints,
                                              CameraParam cp, Mat rVec, Mat tVec) {
        mImgMat.fromArray(imagePoints);
        mObjMat.fromArray(objectPoints);

        Mat camMat = cp.getIntrinsicMatrix();
        MatOfDouble distortion = cp.getDistCoeffs();

        // Failed : Iterative Method CV_ITERATIVE
        // No Good (Bounce) : solvePnPRansac(mObjMat, ... tVec, true);
        return solvePnP(mObjMat, mImgMat, camMat, distortion, rVec, tVec, true, CV_EPNP);
    }


    /**
     * Calibration Params
     */
    public void setEyeCenterOfRotationCorrection(INDArray correction) {
        mEyeRLCenterOfRotationCorrection = correction;
    }

    public void setRightEyeKappa(double alpha, double beta) {
        mRightEyeBall.setKappaProfile(alpha, beta);
    }

    public void setLeftEyeKappa(double alpha, double beta) {
        mLeftEyeBall.setKappaProfile(alpha, beta);
    }

    public void setDistanceFactor(double distFactor) {
        mRightEyeBall.setDistanceFactor(distFactor);
        mLeftEyeBall.setDistanceFactor(distFactor);
    }

    ExtrinsicParam getExtrinsicParam() {
        return mCameraParams.getExtrinsicParam();
    }



    /**
     * Accessor & Mutator
     */
    public INDArray getPointOfGazeOfRightEye() {
        return mRightEyeBall.getPointOfGaze();
    }

    public INDArray getPointOfGazeOfLeftEye() {
        return mLeftEyeBall.getPointOfGaze();
    }

    INDArray getPointOfGazeOfBothEye() {
        return Nd4j.hstack(getPointOfGazeOfRightEye(), getPointOfGazeOfLeftEye());
    }

    public INDArray getCorneaCenter() {
        return mRightEyeBall.getCorneaCenter();
    }

    INDArray getCenterOfRotation(boolean isRight) {
        return isRight ?
            mEyeCenterOfRotationInCamera.getColumn(0):
            mEyeCenterOfRotationInCamera.getColumn(1);
    }

    INDArray getIrisCenter(boolean isRight) {
        return isRight ?
            mRightEyeBall.getIrisCenterPos():
            mLeftEyeBall.getIrisCenterPos();
    }

    INDArray getVisualRayVec(boolean isRight) {
        return isRight ?
            mRightEyeBall.getVisualRayVec():
            mLeftEyeBall.getVisualRayVec();
    }

    INDArray getVisualAxisUnitVec(boolean isRight) {
        return isRight ?
            mRightEyeBall.getVisualAxisUnitVec():
            mLeftEyeBall.getVisualAxisUnitVec();
    }

    INDArray getLensToIrisCenterUnitVec(boolean isRight) {
        return isRight ?
            mLensToIrisCenterUnitVec.getColumn(0):
            mLensToIrisCenterUnitVec.getColumn(1);
    }

    INDArray getFaceCoordinateSystemOnCCS() {
        return mFaceCoordinateSystem;
    }


    /**
     * Checker APIs
     */
    public void checkFacePose(INDArray facePoints, INDArray faceDetails) {
        Point[] imgPoints = mMapper.imagePointsFrom(facePoints, faceDetails);
        Point3[] modelPoints = mMapper.get3dModelPoints();
        boolean res = computeFacePosition(imgPoints, modelPoints, mCameraParams, mRotVec, mTransVec);

        if (!res) {
            Log.e(TAG, "NO FACE POSE");
            return;
        }

        mCameraParams.setExtrinsicParam(toExtrinsicMat(mRotVec, mTransVec));
        Log.e(TAG, toStringMat3x3(mCameraParams.getRotationMat()));
    }

}
