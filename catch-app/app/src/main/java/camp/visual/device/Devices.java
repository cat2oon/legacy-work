package camp.visual.device;

public class Devices {

    private static BaseDevice instance = null;

    public static BaseDevice getDevice() {
        if (instance == null) {
            return null;
        }

        return instance;
    }

    public static void setDevice(BaseDevice device) {
        instance = device;
    }



    //
    // Device List
    //
    public static BaseDevice getNexus5() {
        return new Nexus5();
    }

    public static BaseDevice getGalaxyS8() {
        return new GalaxyS8();
    }

}
