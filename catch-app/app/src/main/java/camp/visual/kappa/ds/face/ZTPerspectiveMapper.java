package camp.visual.kappa.ds.face;

import org.eclipse.collections.impl.list.mutable.FastList;
import org.eclipse.collections.impl.map.mutable.UnifiedMap;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.opencv.core.Point;
import org.opencv.core.Point3;

import static camp.visual.ac.Maths.toColVector3d;
import static camp.visual.kappa.ds.face.FaceKeyPoint.*;

/**
 * ZT to VC-Face
 * https://docs.google.com/spreadsheets/d/1_KBs2xPjTafyGgIoxxsq4AKBtilOSOC3fNoAKRcs2bY/edit#gid=0 (not public)
 *
 * TODO: Rename (FacePerspectiveMapper?)
 */
public class ZTPerspectiveMapper {

    private Face3dModel mFace3dModel;
    private UnifiedMap<FaceKeyPoint, Integer> mKeyPointToImageIdx;


    public ZTPerspectiveMapper() {
        mFace3dModel = new Face3dModel();
        mKeyPointToImageIdx = makeImageMappingTable();
    }

    private UnifiedMap makeImageMappingTable() {
        UnifiedMap table = UnifiedMap.newMap(FaceKeyPoint.getNumOfKeysForPnP());

        table.put(E1, 35);
        table.put(E2, 37);
        table.put(E3, 64);
        table.put(E4, 67);
        table.put(E5, 40);
        table.put(E6, 38);
        table.put(E7, 71);
        table.put(E8, 68);

        table.put(R1, 52);
        table.put(R2, 55);
        table.put(R3, 72);
        table.put(R4, 73);
        table.put(R5, 61);
        table.put(R6, 58);
        table.put(R7, 75);
        table.put(R8, 76);

        table.put(N1,  78);
        table.put(N2,  79);
        table.put(N3,  43);
        table.put(N4,  44);
        table.put(N5,  45);
        table.put(N6,  46);
        table.put(N7,  49);
        table.put(N8,  80);
        table.put(N9,  82);
        table.put(N10, 47);
        table.put(N11, 81);
        table.put(N12, 83);
        table.put(N13, 51);

        table.put(M1, 84);
        table.put(M2, 90);
        table.put(M3, 87);
        table.put(M4, 98);
        table.put(M5, 102);
        table.put(M6, 93);
        table.put(M7, 89);

        table.put(O1, 2);
        table.put(O2, 7);
        table.put(O3, 11);
        table.put(O4, 16);
        table.put(O5, 21);
        table.put(O6, 25);
        table.put(O7, 30);

        return table;
    }


    /**
     * APIs
     */
    public Point[] imagePointsFrom(final INDArray facePoints) {
        FastList<FaceKeyPoint> keys = getKeysForPnP();
        Point[] points = new Point[keys.size()];

        for (int i=0; i<keys.size(); i++) {
            int imgIdx = mKeyPointToImageIdx.get(keys.get(i));
            points[i] = makePointFrom(facePoints, imgIdx);
        }

        return points;
    }

    public Point[] imagePointsFrom(INDArray facePoints, INDArray faceDetails) {
        Point[] points = imagePointsFrom(facePoints);

        // Detail Landmark 대체
        // R1, R2, R5, R6
        points[FaceKeyPoint.getIndexOf(R1)] = makePointFrom(faceDetails, 11);
        points[FaceKeyPoint.getIndexOf(R2)] = makePointFrom(faceDetails, 10);
        points[FaceKeyPoint.getIndexOf(R5)] = makePointFrom(faceDetails, 33);
        points[FaceKeyPoint.getIndexOf(R6)] = makePointFrom(faceDetails, 32);

        // E1 ~ E8 (ZT Basic idx, ZT Detail idx)
        // E1(35, 47), E2(37, 50)            E6(38, 63), E5(40, 60)
        // E3(64, 51), E4(67, 56)            E8(68, 69), E7(71, 64)
        points[FaceKeyPoint.getIndexOf(E1)] = makePointFrom(faceDetails, 47);
        points[FaceKeyPoint.getIndexOf(E2)] = makePointFrom(faceDetails, 50);
        points[FaceKeyPoint.getIndexOf(E3)] = makePointFrom(faceDetails, 51);
        points[FaceKeyPoint.getIndexOf(E4)] = makePointFrom(faceDetails, 56);
        points[FaceKeyPoint.getIndexOf(E5)] = makePointFrom(faceDetails, 60);
        points[FaceKeyPoint.getIndexOf(E6)] = makePointFrom(faceDetails, 63);
        points[FaceKeyPoint.getIndexOf(E7)] = makePointFrom(faceDetails, 64);
        points[FaceKeyPoint.getIndexOf(E8)] = makePointFrom(faceDetails, 69);

        // M1, M2, M7 (ZT Basic idx, ZT Detail idx)
        //          M7(89, 83)
        // M1(84, 70), M2(90, 86)
        points[FaceKeyPoint.getIndexOf(M1)] = makePointFrom(faceDetails, 70);
        points[FaceKeyPoint.getIndexOf(M2)] = makePointFrom(faceDetails, 86);
        points[FaceKeyPoint.getIndexOf(M7)] = makePointFrom(faceDetails, 83);

        return points;
    }

    public INDArray carunclesFrom(final INDArray faceDetails) {
        INDArray CUR = faceDetails.getColumn(10);      // subject 기준 오른쪽
        INDArray CUL = faceDetails.getColumn(32);      // subject 기준 왼쪽
        return Nd4j.hstack(CUR, CUL);
    }

    public Point3[] get3dModelPoints() {
        return mFace3dModel.getModel3dPoints();
    }



    //
    // Helpers
    //
    private Point makePointFrom(INDArray points, int idx) {
        double x = points.getDouble(0, idx);
        double y = points.getDouble(1, idx);
        return new Point(x, y);
    }


    //
    // Delegator
    //
    public INDArray getEyeRotationCenterPointsOfModel(boolean isRight) {
        Point3 p = mFace3dModel.getEyeRotationCenterPoints(isRight);
        return toColVector3d(p.x, p.y, p.z);
    }

    public INDArray getEyeRotationCenterPointsOfModel() {
        INDArray R = getEyeRotationCenterPointsOfModel(true);
        INDArray L = getEyeRotationCenterPointsOfModel(false);
        return Nd4j.hstack(R, L);
    }

    public INDArray getEyeCarunclesPointsOfModel(boolean isRight) {
        Point3 p = mFace3dModel.getEyeCarunclesPointsOfModel(isRight);
        return toColVector3d(p.x, p.y, p.z);
    }

    public INDArray getEyeCarunclesPointsOfModel() {
        INDArray R = getEyeCarunclesPointsOfModel(true);
        INDArray L = getEyeCarunclesPointsOfModel(false);
        return Nd4j.hstack(R, L);
    }

}
