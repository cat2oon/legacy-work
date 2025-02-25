package camp.visual.kappa.zt;

import com.sensetime.stmobile.model.STPoint;

import org.nd4j.linalg.api.ndarray.INDArray;

import camp.visual.kappa.cali.Snapshot;
import camp.visual.kappa.ds.face.ZTPerspectiveMapper;

import static camp.visual.ac.Maths.toMatrixColVec2d;
import static camp.visual.ac.Maths.toZeroMatrix;
import static camp.visual.kappa.optimize.GazeOptimizer.makeAnchorCorrection;

public class ZTDetectionParser {

    public static void intoNDArray(STPoint[] points, INDArray ndArr) {
        for (int i = 0; i < points.length; i++) {
            ndArr.put(0, i, (double) points[i].getX());
            ndArr.put(1, i, (double) points[i].getY());
        }
    }

    public static INDArray toNDArray(STPoint[] points) {
        INDArray ndArr = toZeroMatrix(2, points.length);
        intoNDArray(points, ndArr);
        return ndArr;
    }

    public static INDArray toIrisCenters(INDArray irisPoints) {
        int num_points = (int) irisPoints.shape()[1];
        int num_points_each = num_points / 2;

        double lx = 0;
        double ly = 0;
        double rx = 0;
        double ry = 0;

        // ZT의 경우 LEFT[N/2] + RIGHT[N/2]로 구성됨
        // K-모델은 RL 순서를 따라 변환함
        for (int i=0; i<num_points_each; i++) {
            lx += irisPoints.getDouble(0, i);
            ly += irisPoints.getDouble(1, i);
        }

        for (int i=num_points_each; i<num_points; i++) {
            rx += irisPoints.getDouble(0, i);
            ry += irisPoints.getDouble(1, i);
        }

        lx /= num_points_each;
        ly /= num_points_each;
        rx /= num_points_each;
        ry /= num_points_each;

        return toMatrixColVec2d(lx, ly, rx, ry);
    }

}
