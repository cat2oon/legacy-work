package camp.visual.kappa.ds.face;

import org.eclipse.collections.impl.list.mutable.FastList;

public enum FaceKeyPoint {

    /**
     * 눈썹 (E) (100~199)
     */
    E1(101),
    E2(102),
    E3(103),
    E4(104),
    E5(105),
    E6(106),
    E7(107),
    E8(108),


    /**
     * 눈꼬리 (R) (200~299)
     */
    R1(201),
    R2(202),
    R3(203),
    R4(204),
    R5(205),
    R6(206),
    R7(207),
    R8(208),


    /**
     * 코라인 (N) (300~399)
     */
    N1(301),
    N2(302),
    N3(303),
    N4(304),
    N5(305),
    N6(306),
    N7(307),
    N8(308),
    N9(309),
    N10(310),
    N11(311),
    N12(312),
    N13(313),


    /**
     * 입라인 (M) (400~499)
     */
    M1(401),
    M2(402),
    M3(403),
    M4(404),
    M5(405),
    M6(406),
    M7(407),


    /**
     * 턱 윤곽 (O) (500~599)
     */
    O1(501),
    O2(502),
    O3(503),
    O4(504),
    O5(505),
    O6(506),
    O7(507),


    /**
     * Special Purpose & Debug (0~99)
     */
    MODEL_ORIGIN(0),
    EYE_CENTER_OF_ROTATION_LEFT(10),
    EYE_CENTER_OF_ROTATION_RIGHT(11);


    private int mUID;
    private final static FastList<FaceKeyPoint> KEYS_FOR_PNP;


    static {
        KEYS_FOR_PNP = FastList.newListWith(getOrderedKeysForPnP());
    }

    FaceKeyPoint(int uid) {
        this.mUID = uid;
    }

    private static FaceKeyPoint[] getOrderedKeysForPnP() {
        /* 전체 얼굴 랜드마크
        return new FaceKeyPoint[]{
            E1, E2, E3, E4, E5, E6, E7, E8,
            R1, R2, R3, R4, R5, R6, R7, R8,
            N1, N2, N3, N4, N5, N6, N7, N8, N9, N10, N11, N12, N13,
            M1, M2, M3, M4, M5, M6, M7,
            O1, O2, O3, O4, O5, O6, O7
        };
        */

        // 눈 상하 관련 제거 버전
        return new FaceKeyPoint[]{
            E1, E2, E3, E4, E5, E6, E7, E8,
            R1, R2, // 모델 기준 오른쪽 눈 바깥, 안쪽
            R5, R6, // 모델 기준 왼쪽눈 바깥, 안쪽
            N1, N2, N3, N4, N5, N6, N7, N8, N9, N10, N11, N12, N13,
            M1, M2, M7, // M3, M4, M5, M6,
            O1, O2, O3, O4, O5, O6, O7
        };

        // 눈 상하 제거 및 눈썹 제거 버전
        /*
        return new FaceKeyPoint[]{
            R1, R2, // 모델 기준 오른쪽 눈 바깥, 안쪽
            R5, R6, // 모델 기준 왼쪽눈 바깥, 안쪽
            N1, N2, N3, N4, N5, N6, N7, N8, N9, N10, N11, N12, N13,
            M1, M2,
            O1, O2, O3, O4, O5, O6, O7
        };
        */

        /*
        // 눈 라인 모두 제거 버전
        return new FaceKeyPoint[]{
            E1, E2, E3, E4, E5, E6, E7, E8,
            N1, N2, N3, N4, N5, N6, N7, N8, N9, N10, N11, N12, N13,
            M1, M2, M3, M4, M5, M6, M7,
            O1, O2, O3, O4, O5, O6, O7
        };
        */
    }


    /**
     * APIs
     */
    public static int getIndexOf(FaceKeyPoint keyPoint) {
        return KEYS_FOR_PNP.indexOf(keyPoint);
    }

    public static int getNumOfKeysForPnP() {
        return KEYS_FOR_PNP.size();
    }

    public static FastList<FaceKeyPoint> getKeysForPnP() {
        return KEYS_FOR_PNP;
    }

}
