package camp.visual.kappa.ds.face;

import org.eclipse.collections.impl.list.mutable.FastList;
import org.eclipse.collections.impl.map.mutable.UnifiedMap;
import org.opencv.core.Point3;

import static camp.visual.kappa.ds.face.FaceKeyPoint.*;

public class Face3dModel {

    private Point3[] m3dPointsForPnP;
    private UnifiedMap<FaceKeyPoint, Point3> mKeyPointTo3dPoints;


    public Face3dModel() {
        mKeyPointTo3dPoints = makeMappingTable();
        m3dPointsForPnP = makeOrdered3dPointsForPnP();
    }

    private UnifiedMap makeMappingTable() {
        UnifiedMap table = UnifiedMap.newMap();

        //
        // Points for PnP
        //
        table.put(M1,  new Point3(-22.0, -35.0, -26.5));
        table.put(M2,  new Point3(+22.0, -35.0, -26.5));
        table.put(M3,  new Point3( +0.0, -28.0, -13.5));
        table.put(M4,  new Point3( +0.0, -34.0, -15.5));
        table.put(M5,  new Point3( +0.0, -37.0, -16.5));
        table.put(M6,  new Point3( +0.0, -42.0, -17.0));
        table.put(M7,  new Point3(+12.0, -30.0, -16.5));

        table.put(N1,  new Point3( -8.5, +33.0, -22.0));
        table.put(N2,  new Point3( +8.5, +33.0, -22.0));
        table.put(N3,  new Point3( +0.0, +35.0, -16.5));
        table.put(N4,  new Point3( +0.0, +22.0, -12.0));
        table.put(N5,  new Point3( +0.0, +12.0, - 8.0));
        table.put(N6,  new Point3( +0.0, + 1.0, - 2.0));
        table.put(N7,  new Point3( +0.0, -12.5, -11.0));
        table.put(N8,  new Point3(-15.5, + 5.0, -19.0));
        table.put(N9,  new Point3(-16.0, - 2.0, -15.0));
        table.put(N10, new Point3(-14.0, - 9.0, -17.0));
        table.put(N11, new Point3(+15.5, + 5.0, -19.0));
        table.put(N12, new Point3(+16.0, - 2.0, -15.0));
        table.put(N13, new Point3(+14.0, - 9.0, -17.0));

        table.put(O1,  new Point3(-61.0, +13.0, -43.0));
        table.put(O2,  new Point3(-51.5, -32.0, -41.0));
        table.put(O3,  new Point3(-39.0, -53.0, -35.0));
        table.put(O4,  new Point3(+ 0.0, -77.5, -28.5));
        table.put(O5,  new Point3(+39.0, -53.0, -35.0));
        table.put(O6,  new Point3(+51.5, -32.0, -41.0));
        table.put(O7,  new Point3(+61.0, +13.0, -43.0));

        table.put(E1,  new Point3(-30.5, +57.5, -21.0));
        table.put(E2,  new Point3(-12.0, +55.0, -16.5));
        table.put(E3,  new Point3(-38.0, +52.5, -24.0));
        table.put(E4,  new Point3(-12.0, +51.0, -16.5));
        table.put(E5,  new Point3(+30.5, +57.5, -21.0));
        table.put(E6,  new Point3(+12.0, +55.0, -16.5));
        table.put(E7,  new Point3(+38.0, +52.5, -24.0));
        table.put(E8,  new Point3(+12.0, +51.0, -16.5));

        table.put(R1,  new Point3(-41.0, +36.5, -28.5));
        table.put(R2,  new Point3(-21.5, +33.0, -27.0));
        table.put(R3,  new Point3(-30.5, +38.5, -27.5));
        table.put(R4,  new Point3(-31.0, +30.0, -27.0));
        table.put(R5,  new Point3(+41.0, +36.5, -28.5));
        table.put(R6,  new Point3(+21.5, +33.0, -27.0));
        table.put(R7,  new Point3(+30.5, +38.5, -27.5));
        table.put(R8,  new Point3(+31.0, +30.0, -27.0));


        //
        // Special Purpose
        //
        table.put(MODEL_ORIGIN,  new Point3(0.0, 0.0, 0.0));
        table.put(EYE_CENTER_OF_ROTATION_LEFT,  new Point3(+33.0, +35.0, -33.0));
        table.put(EYE_CENTER_OF_ROTATION_RIGHT, new Point3(-33.0, +35.0, -33.0));

        // invertXAxis(table);

        return table;
    }



    /**
     * APIs
     */
    private Point3[] makeOrdered3dPointsForPnP() {
        FastList<FaceKeyPoint> keys = getKeysForPnP();

        Point3[] points = new Point3[keys.size()];
        for (int i=0; i<keys.size(); i++) {
            points[i] = mKeyPointTo3dPoints.get(keys.get(i));
        }

        return points;
    }

    public Point3[] getModel3dPoints() {
        return m3dPointsForPnP;
    }

    public Point3 getEyeRotationCenterPoints(boolean isRight) {
        FaceKeyPoint key = isRight ? EYE_CENTER_OF_ROTATION_RIGHT : EYE_CENTER_OF_ROTATION_LEFT;
        return mKeyPointTo3dPoints.get(key);
    }

    public Point3 getEyeCarunclesPointsOfModel(boolean isRight) {
        FaceKeyPoint key = isRight ? R2: R6;
        return mKeyPointTo3dPoints.get(key);
    }


    //
    // Helpers
    //
    private void invertXAxis(UnifiedMap<FaceKeyPoint, Point3> points) {
        for (Point3 p : points.values()) {
            p.x *= -1;
        }
    }

    private void invertZAxis(UnifiedMap<FaceKeyPoint, Point3> points) {
        for (Point3 p : points.values()) {
            p.z *= -1;
        }
    }

}
