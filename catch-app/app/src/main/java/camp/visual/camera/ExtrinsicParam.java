package camp.visual.camera;

import org.nd4j.linalg.api.ndarray.INDArray;

import static camp.visual.ac.Maths.muliToColIn;
import static camp.visual.ac.Maths.muliToElementIn;
import static camp.visual.ac.Maths.muliToRowIn;
import static camp.visual.ac.Maths.toColVector;
import static camp.visual.ac.Maths.toMatrix3x3ByRowOrder;

/**
 * TODO: CameraParam에 통합
 * Reference.
 * - docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
 */
public class ExtrinsicParam {

    private INDArray mRMat;
    private INDArray mTVec;

    public ExtrinsicParam(double[] rMat, double[] tVec) {
        mRMat = toMatrix3x3ByRowOrder(rMat);                    // rMat 행단위 전달됨
        mTVec = toColVector(tVec);
        adjustAxis();
    }

    public ExtrinsicParam(INDArray rMat, INDArray tVec) {
        mRMat = rMat;
        mTVec = tVec;
    }


    private void adjustAxis() {
        // Image Coordinate <--> World Coordinate (Y축 반전)
        muliToRowIn(mRMat, 1, -1);          // 회전 행렬 y열 (y축 반전)
        muliToElementIn(mTVec, 1, 0, -1);   // 전치 벡터 y축 반전
        // muliToRowIn(mRMat, 0, -1);       // 이미지 거울상 원복 (불필요)
    }

    public INDArray getRotationMat() {
        return mRMat;
    }

    public INDArray getTranslationVec() {
        return mTVec;
    }


    static ExtrinsicParam from(ExtrinsicParam ep) {
        if (ep == null) {
            return null;
        }

        return new ExtrinsicParam(ep.mRMat.dup(), ep.mTVec.dup());
    }

}
