package camp.visual.kappa.al;

import android.util.Log;

import org.nd4j.linalg.api.ndarray.INDArray;

import camp.visual.ac.Maths;

import static camp.visual.ac.Maths.angleBetweenInPlaneNormal;
import static camp.visual.ac.Maths.calcRotationMatFromOpticAxis;
import static camp.visual.ac.Maths.computeSphereRayIntersection;
import static camp.visual.ac.Maths.cos;
import static camp.visual.ac.Maths.l2normalize;
import static camp.visual.ac.Maths.projAOntoB;
import static camp.visual.ac.Maths.sin;
import static camp.visual.ac.Maths.toColVector3d;
import static camp.visual.ac.Maths.toRowVector3d;
import static camp.visual.ac.Maths.toStringColVec;
import static camp.visual.ac.Maths.toStringMat3x3;
import static camp.visual.ac.Maths.toStringPoint;

/**
 * @Terminology and Abbr
 * <p>
 * (E) CenterOfRotation (안구 회전 중심 : Eyeball sphere 중심)
 * (C) CenterOfCorneaCurvature (각막 곡률 중심 : Corneal sphere 중심, 시축 원점)
 * (L) Corneal Limbus
 * <p>
 * ★ Coordinate System
 * ★ 모두 오른손 좌표계 기준
 * (ECS) Eye Coordinate System
 * (FCS) Face Coordinate System
 * (CCS) Camera Coordinate System (device 기준 좌표계)
 * <p>
 * (LSP) Listing's plane (schorlab.berkeley.edu/vilis/QuaternionLL.htm)
 * (RT) EtoCRotateMatrix - ECS to CCS rotation matrix
 * - 이 회전 변환 행렬은 직교 행렬이므로 A^T == A^-1의 성질을 가짐
 * @Vectors (V, VIS) VisualAxisUnitVector - CCS에서 정의된 시축 벡터
 * (O, OPT) OpticalAxisUnitVector - CCS에서 정의된 안축 벡터
 * (K) KappaUnitVector (K-Vector) - ECS에서 정의된 시축 벡터
 * (OIC) LensToIrisCenterUnitVector - CCS에서 정의된 원점에서 홍채 중심을 향한 벡터
 * @Orientation (theta)  yaw, horizontal
 * (phi)    pitch, vertical
 * (lambda) roll
 * @Kappa (alpha) yaw, horizontal  (radian)
 * (beta)  pitch, vertical  (radian)
 * @PointOfGaze { x, y, z }  - z는 보통 CCS 기준 0
 * @Reference listing's law, donder's law - (schorlab.berkeley.edu/vilis)
 * algorithms - Moshe Eizenman(Prof), Elias Daniel Guestrin(PhD) (k-UZogYAAAAJ / GPKMWIwAAAAJ)
 * Camera Extrinsic, Intrinsic - ksimek.github.io/2012/08/14/decompose/
 * Camera - docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
 */


/**
 * TODO : 모든 수치 연산에 대해 2e-17 매우 작은 값 0 처리 할 것
 * TODO : Optical Vector 경우 교점 계산식에 UnitVec을 요구하므로 단위 벡터를 사용해야 하지만
 *        GazeVector 계산시에는 단위 벡터가 아닌 좀 더 큰 수치를 사용해보자
 * TODO:  Optical PoG만으로 dx, dy, dz를 최적화 하고 kappa 최적화 해보자
 */
public class EyeBall {
    private String TAG = "EyeBall";


    /**
     * Status
     */
    private boolean mIsRightEye;
    private double mDistFactor = 1.0;

    private INDArray mPoG;                      // 시선 응시점
    private INDArray mKappaAngles;              // 카파 각도 (alpha, beta)
    private INDArray mEyeOrientation;           // 안구 정위 (theta, phi, lambda)
    private INDArray mKappaUnitVector;          // 카파 단위 방향 벡터 3D
    private INDArray mVisualRayVector;          // 시선에 거리 mu 적용한 Ray 벡터
    private INDArray mVisualAxisUnitVec;        // 시선 단위 벡터
    private INDArray mEtoCRotationMatrix;       // ECS to CCS 좌표계 변환 직교 행렬
    private INDArray mOpticalAxisUnitVec;       // 안축 단위 벡터 (CCS 기준)
    private INDArray mCenterOfRotation;         // 안구 회전 중심점 (E)
    private INDArray mIrisCenterPos;            // 홍채 중심점 (TEMP)
    private INDArray mCenterOfCorneaCurvature;  // 각막 곡률 중심점 (C)


    /**
     * Stats (unit: mm)
     *
     * Reference
     * - The Wiley Handbook of Human Computer Interaction Set
     */
    public static double mKappaAlpha = 0.0;                // 수직, 수평 성분 통계 없음
    public static double mKappaBeta = 0.0;                 // 수직, 수평 성분 통계 없음
    public static final double mDistanceCE = 5.7;
    public static final double mEyeBallRadius = 12.1;      // 10.5 ~ 13.5mm  (mean 12.1)


    EyeBall(boolean isRightEye) {
        mIsRightEye = isRightEye;
        setKappaProfile(mKappaAlpha, mKappaBeta);
    }


    /**
     * Core APIs
     */
    void updateReferenceFace(Face face) {
        mCenterOfRotation = face.getCenterOfRotation(mIsRightEye);
        INDArray lensToIrisUnitVec = face.getLensToIrisCenterUnitVec(mIsRightEye);
        INDArray irisCenter = computeIrisCenterPos(mCenterOfRotation, lensToIrisUnitVec, mEyeBallRadius);

        if (irisCenter == null) {
            Log.e("***", "Compute iris center failed");
            return;
        }

        mIrisCenterPos = irisCenter;
        mOpticalAxisUnitVec = computeOpticalAxisUnitVec(mCenterOfRotation, irisCenter);

        // Non Approximation Version
        // updateCenterOfCorneaCurvature(mOpticalAxisUnitVec);
        // updateECS(face, mOpticalAxisUnitVec);
        // computePointOfGaze();

        // Approximation Version
        updateECS(face, mOpticalAxisUnitVec);
        mVisualAxisUnitVec = toApproxVisualAxisUnitVec(mEtoCRotationMatrix, mEyeOrientation, mKappaAngles);
        double distance = computeRayDistance(mCenterOfRotation, mVisualAxisUnitVec);
        mVisualRayVector = mVisualAxisUnitVec.mul(distance);
        mPoG = mCenterOfRotation.add(mVisualRayVector);
    }

    public INDArray computePointOfGaze() {
        mVisualAxisUnitVec = toVisualAxisUnitVecFromKappaVec(mEtoCRotationMatrix, mKappaUnitVector);
        double distance = computeRayDistance(mCenterOfCorneaCurvature, mVisualAxisUnitVec) * mDistFactor;
        mVisualRayVector = mVisualAxisUnitVec.mul(distance);
        mPoG = mCenterOfCorneaCurvature.add(mVisualRayVector);

        return mPoG;
    }

    public INDArray approximatePointOfGaze() {
        mVisualAxisUnitVec = toApproxVisualAxisUnitVec(mEtoCRotationMatrix, mEyeOrientation, mKappaAngles);
        double distance = computeRayDistance(mCenterOfRotation, mVisualAxisUnitVec);
        mVisualRayVector = mVisualAxisUnitVec.mul(distance);
        mPoG = mCenterOfRotation.add(mVisualRayVector);

        printItem();

        return mPoG;
    }

    private void updateCenterOfCorneaCurvature(INDArray opticalAxisUnitVec) {
        mCenterOfCorneaCurvature = computeCenterOfCorneaCurvature(mDistanceCE, mCenterOfRotation, opticalAxisUnitVec);
    }

    private void updateECS(Face face, INDArray opticalAxisUnitVec) {
        INDArray fcsOnCCS = face.getFaceCoordinateSystemOnCCS();

        // Non Approximation Version
        // mEtoCRotationMatrix = computeEtoCRotationMatrix(opticalAxisUnitVec, fcsOnCCS);

        // Approximation Version
        mEtoCRotationMatrix = fcsOnCCS;
        mEyeOrientation = computeEyeOrientation(fcsOnCCS, opticalAxisUnitVec);
    }

    private void printItem() {
        Log.e(TAG,
                toStringColVec(mCenterOfRotation) + "  " +
                toStringColVec(mIrisCenterPos) + "  " +
                toStringPoint(mPoG) + "  " +
                toStringColVec(mIrisCenterPos.sub(mCenterOfRotation))
        );
    }



    /**
     * Accessors & Mutators
     */
    public void setDistanceFactor(double distFactor) {
        mDistFactor = distFactor;
    }

    public void setKappaProfile(double alpha, double beta) {
        mKappaAlpha = alpha;
        mKappaBeta = beta;
        mKappaAngles = toRowVector3d(alpha, beta, 0.0);
        mKappaUnitVector = toKappaUnitVecFrom(alpha, beta);
    }

    public INDArray getPointOfGaze() {
        return mPoG;
    }

    public INDArray getVisualRayVec() {
        return mVisualRayVector;
    }

    public INDArray getVisualAxisUnitVec() {
        return mVisualAxisUnitVec;
    }

    public INDArray getIrisCenterPos() {
        return mIrisCenterPos;
    }

    public INDArray getCorneaCenter() {
        return mCenterOfCorneaCurvature;
    }



    /**
     * 점 C에서 출발하는 visual axis 단위 벡터 및 거리 mu 값으로 시선 응시점을 구함.
     * 시축 벡터는 ECS 에서 정의된 K-벡터에 회전 행렬[ECS -> CCS]을 적용하여 구함.
     */
    public static INDArray computePointOfGaze(INDArray centerOfCorneaCurvature,
                                              INDArray ecRotationMatrix,
                                              INDArray kappaUnitVector,
                                              double muRatio) {
        INDArray visualAxisUnitVector = ecRotationMatrix.mmul(kappaUnitVector);
        return computePointOfGaze(centerOfCorneaCurvature, visualAxisUnitVector, muRatio);
    }

    private static INDArray computePointOfGaze(INDArray centerOfCorneaCurvature,
                                               INDArray visualAxisUnitVector,
                                               double muRatio) {
        INDArray visualRayVecOnCCS = visualAxisUnitVector.mul(muRatio);
        return centerOfCorneaCurvature.add(visualRayVecOnCCS);
    }

    static INDArray computePointOfGazeOnCamera(INDArray centerOfCorneaCurvature,
                                               INDArray visualAxisUnitVector) {
        double mu = computeRayDistance(centerOfCorneaCurvature, visualAxisUnitVector);
        return computePointOfGaze(centerOfCorneaCurvature, visualAxisUnitVector, mu);
    }

    public static double computeRayDistance(INDArray centerPos, INDArray rayUnitVec) {
        double cz = centerPos.getDouble(2);
        double vz = rayUnitVec.getDouble(2);
        return -cz / vz;
    }

    /**
     * 점 E를 기준으로 optical axis 단위 벡터와 |EC|의 길이를 받아서 점 C의 위치를 구함.
     */
    public static INDArray computeCenterOfCorneaCurvature(double distanceEC,
                                                          INDArray centerOfRotation,
                                                          INDArray opticalAxisUnitVec) {
        // C = E + vec(EC)
        INDArray rayVector = opticalAxisUnitVec.mul(distanceEC);
        return centerOfRotation.add(rayVector);
    }

    /**
     * ECS to CCS 회전 변환 행렬 구함. ECS Z축은 항상 OpticalAxis 벡터로 함.
     * 1. CCS 기준으로 표현한 Face Coordinate System 좌표축들을 기준으로 둠.
     * 2. FCS_CCS zAxisVec to OpticalAxisVec 회전을 각 좌표축에 적용하여 Z축을 변환함과 동시에
     *    listings' law 및 안구 회전 근육 운동에 따르는 torsion 적용된 X,Y 축을 얻음.
     *    (rodrigues 회전만으로는 실제 torsion 값이 다를 수 있음 추후 고려해 볼 것)
     * 3. CCS to ECS 회전 행렬 생성.
     */
    public static INDArray computeEtoCRotationMatrix(INDArray opticAxisUnitVec,
                                                     INDArray faceCoordinateSystemOnCCS) {
        // 1. 기준 좌표축
        INDArray FCSOnCCS = faceCoordinateSystemOnCCS;

        // 2. Rotation Matrix by Listings' Law
        INDArray ECS = calcRotationMatFromOpticAxis(FCSOnCCS, opticAxisUnitVec);

        return ECS;
    }

    public static INDArray computeEyeOrientation(INDArray FCS, INDArray OPT) {
        // pan (theta) (-Z and projection of vec(OPT) onto XZ-Plane)
        // vec(A) : projection vec(OPT) onto XZ-Plane
        // vec(A) = vec(OPT) - Proj_{y}(vec(OPT))
        // theta = acos(vec(A)@vec(-z) / |A|*|-z|) (FCS 이미 -Z 방향)
        //
        // tilt (phi) (-Z and projection of vec(OPT) onto YZ-Plane)
        // ...

        INDArray x = FCS.getColumn(0);
        INDArray y = FCS.getColumn(1);
        INDArray z = FCS.getColumn(2);

        INDArray vecPan = OPT.sub(projAOntoB(OPT, y));
        double theta = angleBetweenInPlaneNormal(z, vecPan, y);

        INDArray vecTilt = OPT.sub(projAOntoB(OPT, x));
        double phi = angleBetweenInPlaneNormal(vecTilt, z, x);

        return toRowVector3d(theta, phi, 0.0);
    }

    /**
     * 응시점 좌표, 점 C, ECS to CCS 회전 행렬이 주어졌을 때, kappa unit vector (ECS) 구함
     */
    public static INDArray inferKappaUnitVec(INDArray pointOfGaze,
                                             INDArray centerOfCorneaCurvature,
                                             INDArray EtoCRotationMatrix) {
        INDArray RT = EtoCRotationMatrix.transpose();     // CCS to ECS
        INDArray visualAxisVec = pointOfGaze.sub(centerOfCorneaCurvature);
        INDArray visUnitColVec = l2normalize(visualAxisVec).transpose();
        INDArray visECSUnitVec = RT.mmul(visUnitColVec);
        // INDArray visECSUnitColVec = RT.mmul(visUnitColVec).transpose();

        // double visProjX = visECSUnitColVec.getDouble(0);
        // double visProjY = visECSUnitColVec.getDouble(1);
        // double visProjZ = visECSUnitColVec.getDouble(2);

        // double alpha = atan(-1 * visProjX / visProjZ);
        // double beta = asin(visProjY);
        // return toRowVector(new double[] {alpha, beta});

        return visECSUnitVec;
    }

    /**
     * kappa alpha, beta 각으로 ECS (점 C)에서 정의되는 단위 K-벡터 생성
     */
    public static INDArray toKappaUnitVecFrom(double alpha, double beta) {
        return toColVector3d(-sin(alpha) * cos(beta), sin(beta), cos(alpha) * cos(beta));
    }

    /**
     * ECS에서 정의된 KappaUnitVector에 ECS to CCS 변환 행렬을 적용하여 CCS 상의 VisualAxisUnitVector를 생성
     */
    public static INDArray toVisualAxisUnitVecFromKappaVec(INDArray ecRotationMatrix,
                                                           INDArray kappaUnitVector) {
        return ecRotationMatrix.mmul(kappaUnitVector);
    }

    public static INDArray toApproxVisualAxisUnitVec(INDArray FCS,
                                                     INDArray eyeOrientation,
                                                     INDArray kappaAngles) {
        double alpha = kappaAngles.getDouble(0, 0);
        double beta = kappaAngles.getDouble(0, 1);
        return toApproxVisualAxisUnitVec(FCS, eyeOrientation, alpha, beta);
    }

    public static INDArray toApproxVisualAxisUnitVec(INDArray FCS,
                                                     INDArray eyeOrientation,
                                                     double kAlpha, double kBeta) {
        INDArray R = FCS;
        double theta = eyeOrientation.getDouble(0, 0);
        double phi = eyeOrientation.getDouble(0, 1);

        INDArray visualApproxVec = toColVector3d(
            sin(theta + kAlpha) * cos(phi + kBeta),
            sin(phi + kBeta),
            cos(theta + kAlpha) * cos(phi + kBeta)
        );

        INDArray V = R.mmul(visualApproxVec);   // visualApproxVec;
        return V;
    }

    public static INDArray computeOpticalAxisUnitVec(INDArray centerOfRotation,
                                                     INDArray irisCenterPos) {
        INDArray opticalAxisUnitVec = irisCenterPos.sub(centerOfRotation);
        return l2normalize(opticalAxisUnitVec);
    }

    public static INDArray computeIrisCenterPos(INDArray centerOfRotation,
                                                INDArray lensToIrisCenterUnitVec,
                                                double eyeBallRadius) {
        return computeSphereRayIntersection(centerOfRotation, lensToIrisCenterUnitVec, eyeBallRadius);
    }


    /**
     * @Deprecated
     * 안구의 정위를 나타내는 orientation(theta, phi, lambda) 각으로 ECS to CCS
     * 회전 변환 행렬을 생성. 회전만을 표현하기 때문에 ECS, CCS 축 방향만 잘 맞추면 된다.
     */
    static INDArray computeEtoCRotationMatrixFromOrientation(double theta, double phi, double lambda) {
        // Rotation Matrix for ECS to CCS
        // CCS : (Y cross dot Z, 렌즈 기준 위쪽,  렌즈 기준 전방)
        // ECS : (안구 전방 기준 왼쪽, 위쪽, optical axis 방향) [paper 기준]

        // 1. CCS-ECS 반전 (coordinate system 정의에 따라 수정해줄 것)
        INDArray rotFlip = Maths.toMatrix3x3ByRowOrder(-1, 0, 0, 0, 1, 0, 0, 0, -1);

        // 2. eye orientation (theta, X-Z plane)
        INDArray rotTheta = Maths.toMatrix3x3ByRowOrder(cos(theta), 0, -sin(theta), 0, 1, 0, sin(theta), 0, cos(theta));

        // 3. eye orientation (phi, Y-Z plane)
        INDArray rotPhi = Maths.toMatrix3x3ByRowOrder(1, 0, 0, 0, cos(phi), sin(phi), 0, -sin(phi), cos(phi));

        // 4. eye orientation (lambda, Z roll)
        INDArray rotLam = Maths.toMatrix3x3ByRowOrder(cos(lambda), -sin(lambda), 0, sin(lambda), cos(lambda), 0, 0, 0, 1);

        // dot dot dot
        INDArray rotMat = rotFlip.mmul(rotTheta);
        rotMat = rotMat.mmul(rotPhi);
        rotMat = rotMat.mmul(rotLam);

        return rotMat;
    }

    public static INDArray computeEtoCRotationMatrixFromOrientation(INDArray orientation) {
        double theta = orientation.getDouble(0);
        double phi = orientation.getDouble(1);
        double lambda = orientation.getDouble(2);
        return computeEtoCRotationMatrixFromOrientation(theta, phi, lambda);
    }

}
