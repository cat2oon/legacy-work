package camp.visual.device;

import camp.visual.camera.CameraParam;

/**
 * Portrait 기준
 * - Width(x)  : 단축
 * - Height(y) : 장축
 * - Pixel : 이미지 좌표계 방향
 * - Millis : x축은 이미지 좌표계와 동일 / y축은 하늘 방향이 증가
 */
public class BaseDevice {

    // TODO: Resolution ENUM
    private CameraParam mCameraParam;

    private double mPixelPerMillisInWidth;          // 1080.0 / 65.0;
    private double mPixelPerMillisInHeight;         // 1920.0 / 115.0;
    private double mMillisPerPixelInWidth;          // 65.0 / 1080.0;
    private double mMillisPerPixelInHeight;         // 115.0 / 1920.0;

    private double mScreenPixelInWidth;
    private double mScreenPixelInHeight;
    private double mScreenPhysicalMillisInWidth;
    private double mScreenPhysicalMillisInHeight;
    private double mLensToScreenInMillisInWidth;    // -10;
    private double mLensToScreenInMillisInHeight;   // -8.5;


    public BaseDevice() { }

    public void setDeviceSize(double[] screenWHSizeInPixel,
                              double[] physicalScreenWHSizeInMillis,
                              double[] lensToScreenOriginWHInMillis) {
        mScreenPixelInWidth = screenWHSizeInPixel[0];
        mScreenPixelInHeight = screenWHSizeInPixel[1];
        mScreenPhysicalMillisInWidth = physicalScreenWHSizeInMillis[0];
        mScreenPhysicalMillisInHeight = physicalScreenWHSizeInMillis[1];
        mLensToScreenInMillisInWidth = lensToScreenOriginWHInMillis[0];
        mLensToScreenInMillisInHeight = lensToScreenOriginWHInMillis[1];

        mPixelPerMillisInWidth = mScreenPixelInWidth / mScreenPhysicalMillisInWidth;
        mPixelPerMillisInHeight = mScreenPixelInHeight / mScreenPhysicalMillisInHeight;
        mMillisPerPixelInWidth = mScreenPhysicalMillisInWidth / mScreenPixelInWidth;
        mMillisPerPixelInHeight = mScreenPhysicalMillisInHeight / mScreenPixelInHeight;
    }

    public void setCameraParam(CameraParam cameraParam) {
        mCameraParam = cameraParam;
    }



    //
    // Device Screen Size
    //
    public double[] millisFromPixel(double px, double py) {
        return new double[] { px * mMillisPerPixelInWidth, py * mMillisPerPixelInHeight };
    }

    public double[] millisFromPixel(double[] pxy) {
        return millisFromPixel(pxy[0], pxy[1]);
    }

    public double[] pixelFromMillis(double mx, double my) {
        return new double[] { mx * mPixelPerMillisInWidth, my * mPixelPerMillisInHeight};
    }

    public double[] pixelFromMillis(double[] mxy) {
        return pixelFromMillis(mxy[0], mxy[1]);
    }



    public double[] millisFromPixelByLens(double px, double py) {
        double[] xy_mm = millisFromPixel(px, py);
        return new double[] { xy_mm[0] + mLensToScreenInMillisInWidth, -xy_mm[1] + mLensToScreenInMillisInHeight };
    }

    public double[] millisFromPixelByLens(double[] pxy) {
        return millisFromPixelByLens(pxy[0], pxy[1]);
    }

    public double[] pixelFromMillisByLens(double mx, double my) {
        return pixelFromMillis(toPixelWiseMillisFromLensOrigin(mx, my));
    }

    private double[] toPixelWiseMillisFromLensOrigin(double mx, double my) {
        double x_mm = mx - mLensToScreenInMillisInWidth;
        double y_mm = -my + mLensToScreenInMillisInHeight;
        return new double[] { x_mm, y_mm };
    }


    //
    // Device Camera Parameter
    //
    public CameraParam getCameraParamDup() {
        return CameraParam.from(mCameraParam);
    }

}
