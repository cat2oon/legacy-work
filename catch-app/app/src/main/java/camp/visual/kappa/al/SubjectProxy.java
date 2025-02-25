package camp.visual.kappa.al;

import org.nd4j.linalg.api.ndarray.INDArray;

import camp.visual.camera.CameraParam;
import camp.visual.camera.ExtrinsicParam;
import camp.visual.device.Devices;
import camp.visual.kappa.ds.face.ZTPerspectiveMapper;
import camp.visual.kappa.zt.ZTDetectionParser;

import static camp.visual.ac.Maths.toMatrix3x2;
import static camp.visual.kappa.optimize.GazeOptimizer.makeAnchorCorrection;

public class SubjectProxy {

    private Face mFace;
    private static boolean isRightEyeForDebug = true;


    public SubjectProxy() {
        this(Devices.getDevice().getCameraParamDup());
    }

    public SubjectProxy(CameraParam cp) {
        mFace = new Face(cp, new ZTPerspectiveMapper());
    }

    /*
     * Momentum, Activation
     * TODO: 델타 작은 수치 변화 평탄화 momentum, activation 필요할 듯
     * Feature 수준에서 이전 N개의 프레임에 대한 보정을 주도록 해보기
     * Bounce 원인이 되는 부분 찾기 - RMat, TVec 등등.
     * 변경이 작다면 RMat만 계산하고 TVec은 보정값 사용하기 등
     * FaceKeyPoint 변화가 적을 경우 프레임 SKIP 가능
     */


    /**
     * APIs (delegate as proxy)
     */
    public void updateFace(INDArray facePoints, INDArray irisPointsLR, INDArray faceDetails) {
        INDArray irisCenters = ZTDetectionParser.toIrisCenters(irisPointsLR);
        mFace.updateFace(facePoints, irisCenters, faceDetails);
    }

    public void setCalibrationParams(double[] params) {
        double dx = params[0];
        double dy = params[1];
        double dz = params[2];
        double ka = params[3];
        double kb = params[4];

        INDArray eCorrection = makeAnchorCorrection(dx, dy, dz);
        mFace.setEyeCenterOfRotationCorrection(eCorrection);
        mFace.setRightEyeKappa(ka, kb);
        mFace.setLeftEyeKappa(ka, kb);
    }

    public void setDistanceFactor(double distFactor) {
        mFace.setDistanceFactor(distFactor);
    }


    /**
     * Query
     */
    public INDArray getPointOfGazeOfBothEye() {
        return mFace.getPointOfGazeOfBothEye();
    }

    public double[] getPointOfGazeOfRightEye() {
        INDArray PoG = mFace.getPointOfGazeOfRightEye();
        if (PoG == null) {
            return new double[] { 0.0, 0.0 };
        }

        return new double[] {PoG.getDouble(0, 0), PoG.getDouble(1, 0) };
    }

    public INDArray getIrisCenterOfRightEye(boolean isRight) {
        return mFace.getIrisCenter(isRight);
    }

    public double[] getPointOfGazeOfLeftEye() {
        INDArray PoG = mFace.getPointOfGazeOfLeftEye();
        return new double[] {PoG.getDouble(0, 0), PoG.getDouble(1, 0) };
    }

    public INDArray getCenterOfCornea() {
        return mFace.getCorneaCenter();
    }

    public INDArray getCenterOfEyeRotation(boolean isRight) {
        return mFace.getCenterOfRotation(isRight);
    }

    public INDArray getVisualRayVector() {
        return mFace.getVisualRayVec(isRightEyeForDebug);
    }

    public ExtrinsicParam getExtrinsicParam() {
        return mFace.getExtrinsicParam();
    }


    /**
     * Checker APIs
     */
    public void checkFacePose(INDArray facePoints, INDArray faceDetails) {
        mFace.checkFacePose(facePoints, faceDetails);
    }

}
