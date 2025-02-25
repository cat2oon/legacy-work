package camp.visual.ac;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static camp.visual.ac.Maths.acos;
import static camp.visual.ac.Maths.applyRodriguesRotation;
import static camp.visual.ac.Maths.cos;
import static camp.visual.ac.Maths.cross;
import static camp.visual.ac.Maths.getRodriguesParam;
import static camp.visual.ac.Maths.muliToColIn;
import static camp.visual.ac.Maths.muliToElementIn;
import static camp.visual.ac.Maths.muliToRowIn;
import static camp.visual.ac.Maths.rotateCoordinateSystemByNewZAxis;
import static camp.visual.ac.Maths.sin;
import static camp.visual.ac.Maths.toColVector3d;
import static camp.visual.ac.Maths.toMatrix3x2;
import static camp.visual.ac.Maths.toMatrix3x3ByRowOrder;
import static camp.visual.ac.Maths.toRad;
import static camp.visual.ac.Maths.toRowVector3d;
import static camp.visual.ac.Maths.toRowVector4d;
import static camp.visual.ac.Maths.toZeroMatrix;
import static org.hamcrest.CoreMatchers.is;
import static org.junit.Assert.assertThat;

public class MathsTest {

    @Test
    public void testRotateCoordinateSystemByNewZAxis() {
        INDArray CS = toCS(1, 0, 0, 0, 1, 0, 0, 0, 1);
        INDArray newZBasisVec = toRowVector3d(0, 0, -1);
        INDArray expected = toCS(1, 0, 0, 0, 1, 0, 0, 0, -1);
        INDArray actual = rotateCoordinateSystemByNewZAxis(CS, newZBasisVec);

        assertThat(actual, is(expected));
    }

    @Test
    public void testCrossProduct() {
        INDArray x = toRowVector3d(1, 2, 3);
        INDArray y = toRowVector3d(4, 5, 6);
        assertThat(cross(x, y), is(toRowVector3d(-3, 6, -3)));
    }

    @Test
    public void testDotProduct() {
        INDArray x = toRowVector3d(1, 2, 3);
        INDArray y = toRowVector3d(4, 5, 6).transpose();
        assertThat(x.mmul(y).getDouble(0), is(32.0));
    }

    @Test
    public void testGetRodriguesParam() {
        // TODO: { vec_a, vec_b, rot_vec, angle } 만들어서 테스트간 공유

        // x축 회전 90'
        assertRodriguesParam(0, 1, 0, 0, 0, 1, toRP(1, 0, 0, 0));

        // y축 회전 90'
        assertRodriguesParam(0, 0, 1, 1, 0, 0, toRP(0, 1, 0, 0));

        // z축 회전 90'
        assertRodriguesParam(1, 0, 0, 0, 1, 0, toRP(0, 0, 1, 0));

        // x축 회전 45'
        assertRodriguesParam(0, 1, 0, 0, s(45), s(45), toRP(1, 0, 0, toRad(45)));
    }

    @Test
    public void testApplyRodriguesRotation() {
        // x축 회전 90'
        assertApplyRodrigues(0, 1, 0, 1, 0, 0, acos(0), toR3V(0, 0, 1));

        // y축 회전 90'
        assertApplyRodrigues(0, 0, 1, 0, 1, 0, acos(0), toR3V(1, 0, 0));

        // z축 회전 90'
        assertApplyRodrigues(1, 0, 0, 0, 0, 1, acos(0), toR3V(0, 1, 0));

        // x축 회전 45'
        assertApplyRodrigues(0, 1, 0, 1, 0, 0, toRad(45), toR3V(0, s(45), s(45)));
    }

    @Test
    public void testMatrix3x2FromDualColumnVector() {
        INDArray x = toColVector3d(1, 2, 3);
        INDArray y = toColVector3d(4, 5, 6);
        assertThat(Nd4j.hstack(x, y), is(toMatrix3x2(1, 2, 3, 4, 5, 6)));
    }

    @Test
    public void testMatrixAddColumnBroadcast() {
        INDArray a = toColVector3d(-3, -3, -3);
        INDArray o = toMatrix3x2(1, 2, 3, 4, 5, 6);
        assertThat(o.addColumnVector(a), is(toMatrix3x2(-2, -1, 0, 1, 2, 3)));
    }

    @Test
    public void testMatrixAddMatrix() {
        INDArray x = toMatrix3x2( 1, 2,  3, 4,  5, 6);
        INDArray y = toMatrix3x2(-1, 0, -3, 0, -5, 0);
        assertThat(x.add(y), is(toMatrix3x2(0, 2, 0, 4, 0, 6)));
    }

    @Test
    public void testMatrixPutValue() {
        INDArray x = toZeroMatrix(2, 10);
        x.put(0, 0, 0.0);
        x.put(1, 0, 1.0);
        x.put(0, 3, 30.0);
        x.put(1, 3, 31.0);
        x.put(0, 8, 80.0);
        x.put(1, 8, 81.0);

        assertThat(x.getDouble(0, 0), is(0.0));
        assertThat(x.getDouble(1, 0), is(1.0));

        assertThat(x.getDouble(0, 3), is(30.0));
        assertThat(x.getDouble(1, 3), is(31.0));

        assertThat(x.getDouble(0, 8), is(80.0));
        assertThat(x.getDouble(1, 8), is(81.0));
    }

    @Test
    public void testMulElementInPlace() {
        INDArray x = Maths.toMatrix3x3ByRowOrder(
            0, 1, 2,
            3, 4, 5,
            6, 7, 8);

        muliToElementIn(x, 0, 2, -3);
        muliToElementIn(x, 1, 1, 12);
        muliToElementIn(x, 2, 0, 0.5);

        assertThat(x.getDouble(0, 2), is(-6.0));
        assertThat(x.getDouble(1, 1), is(48.0));
        assertThat(x.getDouble(2, 0), is(3.0));
    }

    @Test
    public void testMulInPlace() {
        INDArray x = Maths.toMatrix3x3ByRowOrder(
            0, 1, 2,
            3, 4, 5,
            6, 7, 8);

        muliToColIn(x, 0, -1);
        assertThat(x.getDouble(0, 0), is(-0.0));
        assertThat(x.getDouble(1, 0), is(-3.0));
        assertThat(x.getDouble(2, 0), is(-6.0));

        muliToRowIn(x, 2, 2);
        assertThat(x.getDouble(2, 0), is(-12.0));
        assertThat(x.getDouble(2, 1), is(14.0));
        assertThat(x.getDouble(2, 2), is(16.0));
    }

    @Test
    public void tesToMatrix() {
        INDArray x = toMatrix3x3ByRowOrder(new double[]{ 0, 1, 2, 3, 4, 5, 6, 7, 8 });
        assertThat(x.getDouble(0, 2), is(2.0));
        assertThat(x.getDouble(2, 0), is(6.0));

        INDArray y = toMatrix3x3ByRowOrder(0, 1, 2, 3, 4, 5, 6, 7, 8);
        assertThat(y.getDouble(0, 2), is(2.0));
        assertThat(y.getDouble(2, 0), is(6.0));
    }



    /*
     * helpers
     */
    private void assertRodriguesParam(double ax, double ay, double az,
                                      double bx, double by, double bz, INDArray expected) {
        INDArray from = toRowVector3d(ax, ay, az);
        INDArray to   = toRowVector3d(bx, by, bz);
        assertThat(getRodriguesParam(from, to), is(expected));
    }

    private void assertApplyRodrigues(double vx, double vy, double vz,
                                      double rx, double ry, double rz,
                                      double angle, INDArray expected) {
        INDArray v = toR3V(vx, vy, vz);
        INDArray rodriguesParam = toRP(rx, ry, rz, angle);
        assertThat(applyRodriguesRotation(v, rodriguesParam), is(expected));
    }

    private INDArray toR3V(double x, double y, double z) {
       return toRowVector3d(x, y, z);
    }

    private INDArray toCS(double e11, double e12, double e13,
                          double e21, double e22, double e23,
                          double e31, double e32, double e33) {
        return toMatrix3x3ByRowOrder(new double[]{ e11, e12, e13, e21, e22, e23, e31, e32, e33 });
    }

    private INDArray toRP(double x, double y, double z, double i) {
        return toRowVector4d(x, y, z, i);
    }

    private double s(double degree) {
        return sin(toRad(degree));
    }

    private double c(double degree) {
        return cos(toRad(degree));
    }


}