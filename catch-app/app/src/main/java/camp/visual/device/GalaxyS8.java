package camp.visual.device;

import camp.visual.camera.CameraParam;

public class GalaxyS8 extends BaseDevice {

    public GalaxyS8() {
        setDeviceSize(getScreenWHSizeInPixel(), getPhysicalScreenWHSizeInMillis(), getLensToScreenOriginWHInMillis());
        setCameraParam(get1280x720());
    }


    /**
     * Size
     */
    public static double[] getScreenWHSizeInPixel() {
        // TODO: Device 정보 가져오기
        return new double[] { 1080.0, 2220.0 };
    }

    public static double[] getPhysicalScreenWHSizeInMillis() {
        return new double[] { 65.0, 131 };
    }

    public static double[] getLensToScreenOriginWHInMillis() {
        return new double[] { -46.0, -3.0 };
    }


    public static CameraParam get1280x720() {
        return new CameraParam(962.48, 967.77, 383.5, 622.5, 0.0995, 0.198, 0.0119, 0.00019, 0);
    }

}
