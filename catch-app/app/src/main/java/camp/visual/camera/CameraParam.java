package camp.visual.camera;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;

import static camp.visual.ac.Maths.l2normalize;
import static camp.visual.ac.Maths.toColVector3d;


/**
 * TODO: extrinsicParam 통합
 * Reference.
 * - github.com/opencv/opencv/blob/master/samples/android/camera-calibration/src/org/opencv/samples/
 * cameracalibration/CameraCalibrator.java
 * - mathworks.com/help/vision/ug/camera-calibration.html
 *
 * @Distortion
 * - mathworks.com/products/demos/symbolictlbx/Pixel_location/Camera_Lens_Undistortion.html
 * - pdfs.semanticscholar.org/abd8/45414e9abd76e9e5e0778b007bcb765aa785.pdf
 *   (Rational Radial Distortion Models of Camera Lenses with Analytical Solution for Distortion Correction)
 * - hal-enpc.archives-ouvertes.fr/hal-01556898/document (A Precision analysis of camera distortion models)
 * - Analytically_solving_radial_distortion_parameters.pdf
 */
public class CameraParam {

    private double mFx;
    private double mFy;
    private double mCx;
    private double mCy;
    private double mSkewed = 0;

    private double mK1;
    private double mK2;
    private double mP1;
    private double mP2;
    private double mK3;

    private Mat mIntrinsicMat;
    private ExtrinsicParam mExtrinsicParam;
    private MatOfDouble mEmptyDistortion;
    private MatOfDouble mDistortionCoefficients;


    public CameraParam() { }

    public CameraParam(double fx, double fy, double cx, double cy) {
       this(fx, fy, cx, cy, 0, 0, 0, 0, 0);
    }

    public CameraParam(double fx, double fy, double cx, double cy,
                       double k1, double k2, double p1, double p2, double k3) {
        mFx = fx;
        mFy = fy;
        mCx = cx;
        mCy = cy;
        mK1 = k1;
        mK2 = k2;
        mP1 = p1;
        mP2 = p2;
        mK3 = k3;
        mIntrinsicMat = makeCameraParam(fx, fy, cx, cy);
        mDistortionCoefficients = new MatOfDouble(makeDistortionCoefficients(k1, k2, p1, p2, k3));
        mEmptyDistortion = new MatOfDouble(makeDistortionCoefficients(0, 0, 0, 0, 0));
    }



    /**
     * Maker
     */
    public static CameraParam from(CameraParam p) {
        CameraParam cp = new CameraParam(p.mFx, p.mFy, p.mCx, p.mCy, p.mK1, p.mK2, p.mP1, p.mP2, p.mK3);
        cp.setExtrinsicParam(ExtrinsicParam.from(cp.getExtrinsicParam()));
        return cp;
    }

    private static Mat makeCameraParam(double fx, double fy, double cx, double cy) {
        Mat m = new Mat();
        Mat.eye(3, 3, CvType.CV_64FC1).copyTo(m);

        m.put(0, 0, fx);
        m.put(0, 1, 0.0);
        m.put(0, 2, cx);
        m.put(1, 0, 0.0);
        m.put(1, 1, fy);
        m.put(1, 2, cy);
        m.put(2, 0, 0.0);
        m.put(2, 1, 0.0);
        m.put(2, 2, 1.0);

        return m;
    }

    private static Mat makeDistortionCoefficients(double k1, double k2, double p1, double p2, double k3) {
        Mat m = new Mat();
        Mat.zeros(5, 1, CvType.CV_64FC1).copyTo(m);

        m.put(0, 0, k1);
        m.put(0, 1, k2);
        m.put(0, 2, p1);
        m.put(0, 3, p2);
        m.put(0, 4, k3);

        return m;
    }



    /**
     * APIs
     */
    public Mat getIntrinsicMatrix() {
        return mIntrinsicMat;
    }

    public MatOfDouble getDistCoeffs() {
        return mDistortionCoefficients;
    }

    public MatOfDouble getEmptyDisCoeffs() {
        return mEmptyDistortion;
    }

    public INDArray getUVNormalUnitVecFromPixel(INDArray pixelPoint) {
        // 정규 이미지 공간이므로 Z 거리는 1
        double[] uv = uvNormalize(pixelPoint.getDouble(0), pixelPoint.getDouble(1));
        return l2normalize(toColVector3d(uv[0], uv[1], 1));
    }

    public INDArray getUndistortNormalUnitVec(INDArray pixelPoint) {
        double[] pxy = undistort(pixelPoint.getDouble(0), pixelPoint.getDouble(1));
        return l2normalize(toColVector3d(pxy[0], pxy[1], 1));
    }

    private double[] uvNormalize(double px, double py) {
        double yn = (py - mCy) / mFy;
        double xn = (px - mCx) / mFx - (mSkewed * yn);
        return new double[] { xn, yn };
    }

    private double[] uvDenormalize(double nx, double ny) {
        double px = mFx * (nx + mSkewed*ny) + mCx;
        double py = mFy * ny + mCy;
        return new double[] { px, py };
    }

    private double[] undistort(double px, double py) {
        double[] g = uvNormalize(px, py);

        double[] t = uvNormalize(px, py);
        for (int i=0; i<5; i++) {
            double[] t1 = distort(t[0], t[1]);
            t[0] += 0.8 * (g[0] - t1[0]);
            t[1] += 0.8 * (g[1] - t1[1]);
        }

        return new double[] { t[0], t[1] };
    }

    private double[] distort(double nx, double ny) {
        double r2 = nx * nx + ny * ny;
        double radial = 1 + mK1*r2 + mK2*r2*r2;
        double x = radial * nx + 2*mP1*nx*ny + mP2*(r2 + 2*nx*nx);
        double y = radial * ny + 2*mP2*nx*ny + mP1*(r2 + 2*ny*ny);

        return new double[] { x, y };
    }



    /**
     * Accessor
     */
    public double getCx() {
        return mCx;
    }

    public double getCy() {
        return mCy;
    }

    public void setExtrinsicParam(ExtrinsicParam externalParam) {
        mExtrinsicParam = externalParam;
    }

    public ExtrinsicParam getExtrinsicParam() {
        return mExtrinsicParam;
    }

    public INDArray getRotationMat() {
        return mExtrinsicParam.getRotationMat();
    }

    public INDArray getTranslationVec() {
        return mExtrinsicParam.getTranslationVec();
    }

}
