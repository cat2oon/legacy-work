package camp.visual.ac.cv2;

import org.opencv.core.Mat;

import camp.visual.camera.ExtrinsicParam;

import static org.opencv.calib3d.Calib3d.Rodrigues;

public class CvHelper {

    /**
     * Matrix Helper
     */
    public static double[] fromNativeMat(Mat m) {
        int size = (int) m.total() * m.channels();
        double[] buff = new double[size];
        m.get(0, 0, buff);

        return buff;
    }



    /**
     * Transform Helper
     */
    public static ExtrinsicParam toExtrinsicMat(Mat RVec, Mat TVec) {
        Mat R = new Mat();
        Rodrigues(RVec, R);

        double[] rMat = fromNativeMat(R);
        double[] tVec = fromNativeMat(TVec);
        R.release();

        // Log.e("EP", String.format("tVec : %1.1f %1.1f %1.1f", tVec[0], tVec[1], tVec[2]));

        return new ExtrinsicParam(rMat, tVec);
    }

}
