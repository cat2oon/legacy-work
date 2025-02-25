package camp.visual.kappa.optimize;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import camp.visual.kappa.al.EyeBall;
import camp.visual.camera.CameraParam;
import camp.visual.kappa.ds.face.ZTPerspectiveMapper;

import static camp.visual.ac.Maths.computeSphereRayIntersection;
import static camp.visual.ac.Maths.muliToElementIn;
import static camp.visual.ac.Maths.toMatrix3x2;

public class GazeOptimizer {

    private static ZTPerspectiveMapper mapper = new ZTPerspectiveMapper();
    private static INDArray anchor = mapper.getEyeRotationCenterPointsOfModel();
    private static CameraParam mCP;



    public static double[] evaluate(INDArray R, INDArray T, INDArray IC, double[] delta, double[] kappa) {
        double dx = delta[0];
        double dy = delta[1];
        double dz = delta[2];
        double ka = kappa[0];
        double kb = kappa[1];

        double distCE = EyeBall.mDistanceCE;
        double eyeRadius = EyeBall.mEyeBallRadius;
        INDArray UV = pixelToUnitVec(mCP, IC);


        // # 1~2
        INDArray correction = makeAnchorCorrection(dx, dy, dz);
        INDArray eyePos = correction;       // anchor.add(correction);

        // # 3
        INDArray E = projectWorldToCamera(eyePos, R, T);


        // # 13 select R
        E = E.getColumn(0);
        UV = UV.getColumn(0);

        INDArray I = computeSphereRayIntersection(E, UV, eyeRadius);

        // # 5
        INDArray OPT = EyeBall.computeOpticalAxisUnitVec(E, I);

        // # 6
        INDArray C = EyeBall.computeCenterOfCorneaCurvature(distCE, E, OPT);

        // # 7
        INDArray EtoCRotMat = EyeBall.computeEtoCRotationMatrix(OPT, R);

        // # 8
        INDArray KUnitVec = EyeBall.toKappaUnitVecFrom(ka, kb);

        // # 9
        INDArray VUnitVec = EyeBall.toVisualAxisUnitVecFromKappaVec(EtoCRotMat, KUnitVec);

        // # 10
        INDArray VRay = VUnitVec.mul(EyeBall.computeRayDistance(C, VUnitVec));

        // # 11
        INDArray PoG = C.add(VRay);

        // # 12
        return new double[] { PoG.getDouble(0, 0), PoG.getDouble(1, 0) };
    }


    /*
     * Helper
     */
    public static void setCameraParam(CameraParam cp) {
        mCP = cp;
    }

    public static INDArray makeAnchorCorrection(double eRdx, double eRdy, double eRdz) {
        return toMatrix3x2(eRdx, eRdy, eRdz, -eRdx, eRdy, eRdz);
    }

    public static INDArray pixelToUnitVec(CameraParam cp, INDArray pixel) {
        INDArray R = pixel.getColumn(0);
        INDArray L = pixel.getColumn(1);

        // INDArray uvL = cp.getUVNormalUnitVecFromPixel(L);
        // INDArray uvR = cp.getUVNormalUnitVecFromPixel(R);

        INDArray uvL = cp.getUndistortNormalUnitVec(L);
        INDArray uvR = cp.getUndistortNormalUnitVec(R);

        muliToElementIn(uvL, 1, 0, -1);
        muliToElementIn(uvR, 1, 0, -1);

        return Nd4j.hstack(uvR, uvL);
    }

    public static INDArray projectWorldToCamera(INDArray pos, INDArray R, INDArray T) {
        INDArray rotated = R.mmul(pos);
        INDArray posInCamera = rotated.addColumnVector(T);
        return posInCamera;
    }
}
