package camp.visual.kappa.optimize;

import org.nd4j.linalg.api.ndarray.INDArray;

import camp.visual.kappa.cali.Snapshot;

import static camp.visual.ac.Maths.square;


public class distance {

    public static double distanceL2(Snapshot s, INDArray PoG) {
        double tx = s.getCamX();
        double ty = s.getCamY();
        double px = PoG.getDouble(0, 0);
        double py = PoG.getDouble(1, 0);
        return distanceL2(tx, ty, px, py);
    }

    public static double distanceL2(double[] T, double[] P) {
        return square(T[0] - P[0]) + square(T[1] - P[1]);
    }

    public static double distanceL2(double tx, double ty, double px, double py) {
        return square(tx - px) + square(ty - py);
    }

    public static double distXSquare(Snapshot ss, INDArray PoG) {
        double tx = ss.getCamX();
        double px = PoG.getDouble(0, 0);
        return square(tx - px);
    }

    public static double distYSquare(Snapshot ss, INDArray PoG) {
        double ty = ss.getCamY();
        double py = PoG.getDouble(1, 0);
        return square(ty - py);
    }

}
