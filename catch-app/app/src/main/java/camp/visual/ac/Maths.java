package camp.visual.ac;

import org.apache.commons.lang3.ArrayUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.SpecifiedIndex;

import static org.nd4j.linalg.ops.transforms.Transforms.round;

public class Maths {


    /*
     * Basic
     */
    public static double abs(double x) {
        return Math.abs(x);
    }

    public static double square(double x) {
        return Math.pow(x, 2);
    }

    public static double sqrt(double x) {
        return Math.sqrt(x);
    }


    /*
     * trigonometry
     */
    public static double cos(double x) {
        return Math.cos(x);
    }

    public static double acos(double x) {
        return Math.acos(x);
    }

    public static double sin(double x) {
        return Math.sin(x);
    }

    public static double asin(double x) {
        return Math.asin(x);
    }

    public static double tan(double x) {
        return Math.tan(x);
    }

    public static double atan(double x) {
        return Math.atan(x);
    }

    public static double toDeg(double x) {
        return Math.toDegrees(x);
    }

    public static double toRad(double x) {
        return Math.toRadians(x);
    }


    /*
     * Binary Ops
     */
    public static boolean equal(INDArray x, INDArray y) {
        return ArrayUtils.isEquals(x, y);
    }


    /*
     * NdArray
     */
    public static INDArray toMatrix3x3ByRowOrder(double e11, double e12, double e13,
                                                 double e21, double e22, double e23,
                                                 double e31, double e32, double e33) {
        return Nd4j.create(new double[]{
            e11, e12, e13,
            e21, e22, e23,
            e31, e32, e33,
        }, new int[]{3, 3});
    }

    public static INDArray toMatrix3x3ByRowOrder(double[] mat) {
        return Nd4j.create(mat, new int[]{3, 3});
    }

    public static INDArray toCoordinateSystem(INDArray x, INDArray y, INDArray z) {
        return Nd4j.hstack(x.transpose(), y.transpose(), z.transpose());
    }

    public static INDArray identityMatrix(int n) {
        return Nd4j.eye(n);
    }

    public static INDArray toZeroMatrix(int num_rows, int num_cols) {
        return Nd4j.zeros(num_rows, num_cols);
    }

    public static INDArray toMatrix3x2(double x1, double y1, double z1,
                                       double x2, double y2, double z2) {
        return Nd4j.create(new double[] { x1, x2, y1, y2, z1, z2 }, new int[]{3, 2});
    }

    public static INDArray toMatrixColVec2d(double x1, double y1, double x2, double y2) {
        return Nd4j.create(new double[] { x1, x2, y1, y2 }, new int[]{2, 2});
    }


    /*
     * NdArray - Vector
     */
    public static double angleBetweenTwoUnitVecs(INDArray x, INDArray y) {
        return acos(dot(x, y));
    }

    public static double angleBetweenUnsigned(INDArray x, INDArray y) {
        return acos(dot(l2normalize(x), l2normalize(y)));
    }

    public static double angleBetweenInPlaneNormal(INDArray x, INDArray y, INDArray planeNormal) {
        // stackoverflow.com/a/33920320/11463074
        return Math.atan2(dot(cross(y, x), l2normalize(planeNormal)), dot(x, y));
    }

    public static double dot(INDArray x, INDArray y) {
        if (x.isColumnVector()) {
            x = x.transpose();
        }

        if (y.isRowVector()) {
            y = y.transpose();
        }

        return x.mmul(y).getDouble(0);
    }

    public static INDArray cross(INDArray x, INDArray y) {
        double x0 = x.getDouble(0);
        double x1 = x.getDouble(1);
        double x2 = x.getDouble(2);
        double y0 = y.getDouble(0);
        double y1 = y.getDouble(1);
        double y2 = y.getDouble(2);

        return toRowVector3d((x1*y2)-(x2*y1), (x2*y0)-(x0*y2), (x0*y1)-(x1*y0));
    }

    public static INDArray projAOntoB(INDArray a, INDArray b) {
        return b.mul(dot(a, b) / (square((double) b.norm2Number())));
    }

    public static INDArray l2normalize(INDArray arr) {
        return arr.div(arr.norm2Number());
    }

    public static INDArray toColVector(double[] arr) {
        return Nd4j.create(arr, new int[]{arr.length, 1});
    }

    public static INDArray toColVector3d(double x, double y, double z) {
        return toColVector(new double[] {x, y, z});
    }

    public static INDArray toRowVector(double[] arr) {
        return Nd4j.create(arr, new int[]{1, arr.length});
    }

    public static INDArray toRowVector3d(double x, double y, double z) {
        return toRowVector(new double[] {x, y, z});
    }

    public static INDArray toRowVector4d(double x, double y, double z, double i) {
        return toRowVector(new double[] {x, y, z, i});
    }



    /*
     * NdArray - In-Place Operation
     */
    public static void muliToElementIn(INDArray ndArr, int row, int col, double mul) {
        ndArr.put(row, col, mul * ndArr.getDouble(row, col));
    }

    public static void muliToColIn(INDArray ndArr, int colIdx, double mul) {
        INDArray col = ndArr.getColumn(colIdx);
        ndArr.putColumn(colIdx, col.mul(mul));
    }

    public static void muliToRowIn(INDArray ndArr, int rowIdx, double mul) {
        INDArray col = ndArr.getRow(rowIdx);
        ndArr.putRow(rowIdx, col.mul(mul));
    }



    /*
     * Transform
     */
    public static INDArray roundPrecision(INDArray arr) {
        return round(arr);
    }

    public static INDArray toRotationAxisAndAngleAlong(INDArray fromVec,
                                                       INDArray toVec) {
        toVec = l2normalize(toVec);
        fromVec = l2normalize(fromVec);

        double theta = angleBetweenTwoUnitVecs(fromVec, toVec);
        INDArray rotationAxis = cross(fromVec, toVec);

        double x = rotationAxis.getDouble(0);
        double y = rotationAxis.getDouble(1);
        double z = rotationAxis.getDouble(2);
        double t = 1 - cos(theta);
        double beta = cos(theta);
        double alpha = sin(theta);

        /*
         * mathworld.wolfram.com/RodriguesRotationFormula.html
         */
        double[] rotMat = new double[] {
            (beta) + (x*x*t), (x*y*t) - (z*alpha), (y*alpha) + (x*z*t),
            (z*alpha) + (x*y*t), (beta) + (y*y*t), (-x*alpha) + (y*z*t),
            -(y*alpha) + (x*z*t), (x*alpha) + (y*z*t), (beta) + (z*z*t)
        };

        return toMatrix3x3ByRowOrder(rotMat);
    }

    public static INDArray getRodriguesParam(INDArray fromVec,
                                             INDArray toVec) {
        toVec = l2normalize(toVec);
        fromVec = l2normalize(fromVec);
        double theta = angleBetweenTwoUnitVecs(fromVec, toVec);
        INDArray rotationAxis = cross(fromVec, toVec);

        return Nd4j.hstack(l2normalize(rotationAxis), toRowVector(new double[] {theta}));
    }

    public static INDArray applyRodriguesRotation(INDArray v, INDArray rodriguesParam) {
        double theta = rodriguesParam.getDouble(3);
        INDArray k = rodriguesParam.get(new SpecifiedIndex(0, 1, 2)).reshape(3);

        INDArray r = v.mul(cos(theta));
        r = r.add(cross(k, v).mul(sin(theta)));
        r = r.add(k.mul(dot(k, v) * (1-cos(theta))));

        return r;
    }

    @Deprecated
    public static INDArray rotateCoordinateSystemByNewZAxis(INDArray coordinateSystem,
                                                            INDArray newZAxisVec) {
        /**
         * 결과가 Orthogonal 보장하지 않아서 사용할 수 없음
         */
        if (equal(coordinateSystem.getColumn(2), newZAxisVec)) {
            return coordinateSystem;
        }

        INDArray xAxisVec = coordinateSystem.getColumn(0).transpose();  // Row
        INDArray yAxisVec = coordinateSystem.getColumn(1).transpose();  // Row
        INDArray zAxisVec = coordinateSystem.getColumn(2);              // Col

        INDArray rodParam = getRodriguesParam(zAxisVec, newZAxisVec);
        INDArray newXAxisVec = applyRodriguesRotation(xAxisVec, rodParam);  // Row
        INDArray newYAxisVec = applyRodriguesRotation(yAxisVec, rodParam);  // Row

        return toCoordinateSystem(newXAxisVec, newYAxisVec, newZAxisVec.transpose());   // Col, Col, Col
    }

    public static INDArray calcRotationMatFromOpticAxis(INDArray mat, INDArray opticAxisUnitVec) {
        INDArray OPT = opticAxisUnitVec.mul(-1);            // -Z 방향으로 phi, theta 정의

        /**
         *  Rotation Matrix by Listings' law. R(θ, φ)
         *  Notice. -Z 방향으로 Optical Axis Vec 방향을 기준함
         *          CCS 기준 optic 벡터라면 역방향 취할 것.
         *          단, 리턴값에는 그대로 CCS 기준
         *
         * r11 = 1 - ((sin^2(θ) * cos^2(φ)) / (1 + cos(θ) * cos(φ)))
         * r12 = -sin(φ)*sin(θ)*cos(φ) / (1 + cos(θ)*cos(φ))
         * r13 = -sin(θ)*cos(φ)
         *
         * r21 = -sin(φ)*sin(θ)*cos(φ) / (1 + cos(θ)*cos(φ))
         * r22 =  (cos(θ)*cos(φ) + cos^2(φ)) / (1 + cos(θ)*cos(φ))
         * r23 = -sin(φ)
         *
         * r31 = sin(θ)*cos(φ)
         * r32 = sin(φ)
         * r33 = cos(θ)*cos(φ)
         *
         **/

        double sp = -1.0 * OPT.getDouble(1);                // sin(phi)
        double cpsq = 1.0 - square(sp);                     // cos(phi) * cos(phi) (cos square)
        double ctcp = 1.0 * OPT.getDouble(2);               // cos(theta) * cos(phi)
        double stcp = -1.0 * OPT.getDouble(0);              // sin(theta) * cos(phi)

        double r11 = 1 - (square(stcp) / (1+ctcp));
        double r12 = -sp * stcp / (1+stcp);
        double r21 = -sp*stcp / (1+ctcp);
        double r22 = (ctcp + cpsq) / (1+ctcp);
        double r31 = stcp;
        double r32 = sp;

        return toMatrix3x3ByRowOrder(r11, r12, stcp, r21, r22, sp, r31, r32, -ctcp);
    }


    /*
     * Solver
     */
    public static INDArray computeSphereRayIntersection(INDArray sphereCenter,
                                                        INDArray rayUnitVecFromOrigin,
                                                        double sphereRadius) {
        assert (sphereRadius > 0);

        // return sphereRayIntersectionByAnalytic(sphereCenter, rayUnitVecFromOrigin, sphereRadius);
        return sphereRayIntersectionByGeometric(sphereCenter, rayUnitVecFromOrigin, sphereRadius);
    }

    private static INDArray sphereRayIntersectionByAnalytic(INDArray sphereCenter,
                                                            INDArray rayUnitVecFromOrigin,
                                                            double sphereRadius) {
        double R = sphereRadius;
        INDArray C = sphereCenter;
        INDArray uv = rayUnitVecFromOrigin.transpose();

        /**
         * en.wikipedia.org/wiki/Line–sphere_intersection
         * X = O + (d * uv)
         * Det : (UV @ C)**2 - (|C|**2 - R**2)
         *
         * TODO: 다른 방식의 교점 수식 구하기 (R이 너무 작아 unstable 함)
         * 1. 교점이 없는 경우 z축 기준으로 radius 이동한 위치로 고정
         *
         * Notice. 최초 캘리브레이션 과정에서는 교점이 없더라도 대강의 해를 제공해야
         * 근사 과정을 진행할 수 있음 (심도를 eyeball radius인 11mm로 조정하여 제공)
         */
        double uvDotC = uv.mmul(C).getDouble(0);
        double det = square(uvDotC) - (square((Double) C.norm2Number()) - square(R));
        double magnitude = uvDotC - sqrt(det);      // 교점 중 카메라 기준 가까운 쪽 선택

        if (det < 0) {
            magnitude = uvDotC - 11;
        }

        INDArray intersection = uv.mul(magnitude);
        return intersection.transpose();
    }

    private static INDArray sphereRayIntersectionByGeometric(INDArray sphereCenter,
                                                             INDArray rayUnitVecFromOrigin,
                                                             double sphereRadius) {
        /**
         * Ref. www.scratchapixel.com/lessons/3d-basic-rendering/
         *      minimal-ray-tracer-rendering-simple-shapes/ray-sphere-intersection
         * P0 = O + to * vec(UV)
         * P1 = O + t1 * vec(UV)
         *
         * - A는 vec(UV)와 E가 직교로 만나는 점
         * tca = vec(OA) == vec(E) @ vec(UV)
         */

        double r = sphereRadius;
        INDArray UV = rayUnitVecFromOrigin;
        INDArray E = sphereCenter;

        double tca = E.transpose().mmul(UV).getDouble(0);
        double eNorm = square((double) E.norm2Number());
        double tcaSq = square(tca);

        if (tcaSq > eNorm) {
            return null;
        }

        double d = sqrt(eNorm - tcaSq);
        double thcSq = square(r) - square(d);

        if (thcSq < 0) {
            return null;
        }

        return UV.mul(tca - sqrt(thcSq));
    }


    /*
     * Printer
     */
    public static String toStringMat3x3(INDArray m) {
        return String.format(
            "%1.3f, %1.3f, %1.3f, " +
            "%1.3f, %1.3f, %1.3f, " +
            "%1.3f, %1.3f, %1.3f  ",
                m.getDouble(0, 0), m.getDouble(0, 1), m.getDouble(0, 2),
                m.getDouble(1, 0), m.getDouble(1, 1), m.getDouble(1, 2),
                m.getDouble(2, 0), m.getDouble(2, 1), m.getDouble(2, 2)
        );
    }

    public static String toStringTwoColVec(INDArray m) {
        return String.format(
            "[%1.3f, %1.3f, %1.3f] [%1.3f, %1.3f, %1.3f]",
            m.getDouble(0, 0), m.getDouble(1, 0), m.getDouble(2, 0),
            m.getDouble(0, 1), m.getDouble(1, 1), m.getDouble(2, 1)
        );
    }

    public static String toStringColVec(INDArray v) {
        return String.format("%1.1f, %1.1f, %1.1f",
                v.getDouble(0, 0), v.getDouble(1, 0), v.getDouble(2, 0)
        );
    }

    public static String toStringRowVec(INDArray v) {
        return toStringColVec(v.transpose());
    }

    public static String toStringPoint(INDArray p) {
        if (p.isColumnVector()) {
            return String.format("%1.1f, %1.1f", p.getDouble(0, 0), p.getDouble(1, 0));
        } else {
            return String.format("%1.1f, %1.1f", p.getDouble(0, 0), p.getDouble(0, 1));
        }
    }

}
