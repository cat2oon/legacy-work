package camp.visual.kappa.al;

import org.hamcrest.Matcher;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;

import static camp.visual.ac.Maths.tan;
import static camp.visual.ac.Maths.toColVector;
import static camp.visual.ac.Maths.toRad;
import static camp.visual.ac.Maths.toMatrix3x3ByRowOrder;
import static org.hamcrest.CoreMatchers.is;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThat;

public class EyeBallTest {

    @Test
    public void makeKappaUnitVecFromAngle() {
        // when optical axis is equal to z-axis
        assertKappaWith(0, 0, 0, 0, 1);

        // when alpha == 90'
        assertKappaWith(toRad(90), 0, -1, 0, 0);

        // when  beta == 90'
        assertKappaWith(0, toRad(90), 0, 1, 0);

        // when alpha == 45' and beta == 45'
        assertKappaWith(toRad(45), toRad(45), -0.5, 0.7071, 0.5);
    }


    @Test
    @Ignore("좌표축 기준 정하고 거기에 맞추어 수정해야 함")
    public void computeRotationMatrix() {
        // when eyeball orientation is primary position
        INDArray actual = EyeBall.computeEtoCRotationMatrixFromOrientation(0, 0, 0);
        assertThat(actual, is(toR(-1, 0, 0, 0, 1, 0, 0, 0, -1)));

        // when theta (horizontal) == 90'
        actual = EyeBall.computeEtoCRotationMatrixFromOrientation(toRad(90), 0, 0);
        assertThat(actual, is(toR(-1, 0, 0, 0, 1, 0, 0, 0, 1)));

        // when phi (vertical) == 90'
        actual = EyeBall.computeEtoCRotationMatrixFromOrientation(0, toRad(90), 0);
        assertThat(actual, is(toR(-1, 0, 0, 0, 0, 1, 0, 1, 0)));
    }

    @Test
    public void computePointOfGaze() {
        // 기본 축 회전 변환
        INDArray R = toR(1, 0, 0, 0, -1, 0, 0, 0, -1);

        // 정면에서 정면으로 응시
        assertPOG(toK(0, 0), toC(0, 0, 300), R, is(toPoG(0, 0)));

        // 정면에서 아래로 응시
        assertPOG(toK(0, toRad(15)), toC(0, 0, 300), R, is(toPoG(0, -300 * tan(toRad(15)))));
    }



    /*
     * helpers
     */
    private void assertPOG(INDArray K, INDArray C, INDArray R, Matcher<INDArray> is) {
        INDArray V = EyeBall.toVisualAxisUnitVecFromKappaVec(R, K);
        assertThat(EyeBall.computePointOfGazeOnCamera(C, V), is);
    }

    private void assertKappaWith(double alpha, double beta, double x, double y, double z) {
        INDArray expected = toColVec(x, y, z);
        INDArray actual = EyeBall.toKappaUnitVecFrom(alpha, beta);
        assertEquals(expected, actual);
    }

    private INDArray toColVec(double x, double y, double z) {
        return toColVector(new double[]{x, y, z});
    }

    private INDArray toK(double alpha, double beta) {
        return EyeBall.toKappaUnitVecFrom(alpha, beta);
    }

    private INDArray toC(double x, double y, double z) {
        return toColVec(x, y, z);
    }

    private INDArray toPoG(double x, double y, double z) {
        return toColVec(x, y, z);
    }

    private INDArray toPoG(double x, double y) {
        return toColVec(x, y, 0);
    }

    private INDArray toR(double e11, double e12, double e13,
                         double e21, double e22, double e23,
                         double e31, double e32, double e33) {
        double[] mat = { e11, e12, e13, e21, e22, e23, e31, e32, e33 };
        return toMatrix3x3ByRowOrder(mat);
    }


}