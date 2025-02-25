package camp.visual.kappa.optimize;

import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;

import camp.visual.ac.Maths;
import camp.visual.camera.CameraParam;
import camp.visual.device.BaseDevice;
import camp.visual.device.Devices;

import static camp.visual.ac.Maths.toColVector3d;
import static camp.visual.ac.Maths.toMatrix3x3ByRowOrder;
import static camp.visual.ac.Maths.toMatrixColVec2d;
import static camp.visual.ac.Maths.toRad;
import static camp.visual.kappa.optimize.distance.distanceL2;
import static junit.framework.TestCase.assertTrue;
import static org.hamcrest.CoreMatchers.is;
import static org.hamcrest.MatcherAssert.assertThat;

public class GazeOptimizerTest {

    BaseDevice device;
    INDArray R, T, IC;
    CameraParam cp;
    double[] pXY;
    double[] tXY = new double[2];
    double[] kappa = new double[2];
    double[] delta = new double[3];


    @Before
    public void setup() {
        device = Devices.getNexus5();
        cp = mockCP();
        R = toMirrorR();        // Mirror
        toT(0, 0, 200);         // 20mm
        toScreenCenter();       // IC Center

        GazeOptimizer.setCameraParam(cp);
    }


    @Test
    @Ignore
    public void 정면_미러_평가() {
        evaluate();
        checkPog();
        distLessThan(5);
    }

    @Test
    @Ignore
    public void 정면_미러_평가_ka_3() {
        kappa(-3, 0);
        truth(10, 0);
        evaluate();
        checkPog();
        distLessThan(5);
    }

    @Test
    @Ignore
    public void 정면_미러_평가_kb_3() {
        kappa(0, 3);
        truth(0, 10);
        evaluate();
        checkPog();
        distLessThan(5);
    }

    @Test
    public void 정면_상위응시_미러_평가() {
        truth(0, 10);
        toICFromCenter(0, -10);
        evaluate();
        checkY();
        distLessThan(5);
    }




    /*
     * Helper
     */
    private void checkX() {
        assertThat((int)pXY[0], is((int)tXY[0]));
    }

    private void checkY() {
        assertThat((int)pXY[1], is((int)tXY[1]));
    }

    private void checkPog() {
        assertThat((int)pXY[0], is((int)tXY[0]));
        assertThat((int)pXY[1], is((int)tXY[1]));
    }

    private void distLessThan(double limit) {
        assertTrue(distance() < limit);
    }

    private void evaluate() {
       pXY = GazeOptimizer.evaluate(R, T, IC, delta, kappa);
    }

    private INDArray toMirrorR() {
        return toMatrix3x3ByRowOrder(new double [] {
            1,  0,  0,
            0,  1,  0,
            0,  0, -1,
        });
    }

    private void toT(double x, double y, double z) {
        T = toColVector3d(x, y, z);
    }

    private void toICFromCenter(double mx, double my) {
        double cx = 720 / 2;
        double cy = 1280 / 2;
        double[] mxy = device.pixelFromMillis(mx, my);
        IC = toMatrixColVec2d(cx+mxy[0], cy+mxy[1], cx+mxy[0], cy+mxy[1]);
    }

    private void toScreenCenter() {
        double x = 720/2;
        double y = 1280/2;
        IC = toMatrixColVec2d(x, y, x, y);
    }

    private void truth(double x, double y) {
        tXY[0] = x;
        tXY[1] = y;
    }

    private void delta(double x, double y, double z) {
        delta[0] = x;
        delta[1] = y;
        delta[2] = z;
    }

    private void kappa(double alphaDegree, double betaDegree) {
        kappa[0] = toRad(alphaDegree);
        kappa[1] = toRad(betaDegree);
    }

    private double distance() {
        return distanceL2(tXY, pXY);
    }

    private CameraParam mockCP() {
        CameraParam cp = new CameraParam() {
            public INDArray getUVNormalUnitVecFromPixel(INDArray pixelPoint) {
                double u = (pixelPoint.getDouble(0) - 360) / 950;
                double v = (pixelPoint.getDouble(1) - 640) / 950;
                return Maths.l2normalize(toColVector3d(u, v, 1));
            }
        };

        return cp;
    }
}