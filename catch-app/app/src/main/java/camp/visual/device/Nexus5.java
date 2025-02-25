package camp.visual.device;

import camp.visual.camera.CameraParam;

public class Nexus5 extends BaseDevice {

    public Nexus5() {
        setDeviceSize(
            getScreenWHSizeInPixel(),
            getPhysicalScreenWHSizeInMillis(),
            getLensToScreenOriginWHInMillis());

        setCameraParam(get1280x720());
    }


    public static double[] getScreenWHSizeInPixel() {
        return new double[] { 1080.0, 1920.0 };
    }

    public static double[] getPhysicalScreenWHSizeInMillis() {
        return new double[] { 65.0, 115.0 };
    }

    public static double[] getLensToScreenOriginWHInMillis() {
        return new double[] { -10.0, -8.5 };
    }


    public static CameraParam get640x480() {
        return new CameraParam(950 * 3 / 2, 950 / 2, 480 / 2, 640 / 2);
    }

    public static CameraParam get1280x720() {
        return new CameraParam(950, 950, 720 / 2, 1280 / 2);
    }

}
