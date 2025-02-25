package camp.visual.kappa.cali;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.os.Environment;

import com.sensetime.stmobile.model.STPoint;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;

import camp.visual.camera.CameraParam;

import static camp.visual.kappa.zt.ZTDetectionParser.intoNDArray;
import static camp.visual.kappa.zt.ZTDetectionParser.toIrisCenters;
import static camp.visual.kappa.zt.ZTDetectionParser.toNDArray;

public class Snapshot {

    // Idx
    public int mIdx;
    public boolean mIsDetectionSuccess = true;
    private byte[] mFrameImage;             // 캘리브레이션 당시 촬영 사진

    // 정답 좌표
    private double mCamX;                   // ground truth x (사용자 기준 오른쪽)
    private double mCamY;                   // ground truth y (위쪽)

    // Face 정보
    private STPoint mFacePoints[];
    private STPoint mIrisPoints[];
    private STPoint mFaceDetails[];

    private INDArray mFacePointsNdArr;
    private INDArray mFaceDetailsNdArr;
    private INDArray mIrisCentersPointNdArr;

    // Camera
    private INDArray mE;
    private CameraParam mCP;

    // optimal param
    private double[] mOptimalParam;

    public Snapshot(int frameSize, int idx) {
        mIdx = idx;
        mFrameImage = new byte[frameSize];
        mFacePoints = new STPoint[106];
        mIrisPoints = new STPoint[38];
        mFaceDetails = new STPoint[134];
    }



    //
    // Processed
    //
    public INDArray getE() {
        return mE;
    }

    public void setInitialE(INDArray E) {
        mE = E;
    }



    //
    // Ground Truth
    //
    public double getCamX() {
        return mCamX;
    }

    public double getCamY() {
        return mCamY;
    }

    public void setCamXY(double x, double y) {
        mCamX = x;
        mCamY = y;
    }


    //
    // Image Pixel
    //
    public STPoint[] getFaceDetails() {
        return mFaceDetails;
    }

    public STPoint[] getIrisPoints() {
        return mIrisPoints;
    }

    public STPoint[] getFacePoints() {
        return mFacePoints;
    }

    public INDArray getIrisCentersInPixelNDArr() {
        return mIrisCentersPointNdArr;
    }

    public INDArray getFacePointsNDArr() {
        return mFacePointsNdArr;
    }

    public INDArray getFaceDetailsNDArr() {
        return mFaceDetailsNdArr;
    }

    public void updateFacePoints(STPoint[] facePoints, STPoint[] irisPoints, STPoint[] faceDetails) {
        setFaceInfo(facePoints, irisPoints, faceDetails);
        mFacePointsNdArr = toNDArray(facePoints);
        mFaceDetailsNdArr = toNDArray(faceDetails);
        mIrisCentersPointNdArr = toIrisCenters(toNDArray(irisPoints));
    }

    private void setFaceInfo(STPoint[] facePoints, STPoint[] irisPoints, STPoint[] faceDetails) {
        for (int i = 0; i < facePoints.length; i++) {
            mFacePoints[i] = new STPoint(facePoints[i].getX(), facePoints[i].getY());
        }

        for (int i = 0; i < irisPoints.length; i++) {
            mIrisPoints[i] = new STPoint(irisPoints[i].getX(), irisPoints[i].getY());
        }

        for (int i = 0; i < faceDetails.length; i++) {
            mFaceDetails[i] = new STPoint(faceDetails[i].getX(), faceDetails[i].getY());
        }
    }


    //
    // Camera
    //
    public INDArray getRotationMat() {
        return mCP.getRotationMat();
    }

    public void setCameraParams(CameraParam cp) {
        mCP = cp;
    }

    public CameraParam getCameraParams() {
        return mCP;
    }


    //
    // Miscellaneous
    //
    public void setFaceDetectFail() {
        mIsDetectionSuccess = false;
    }


    //
    // Helper
    //
    private Bitmap toBitmapImage() {
        return decodeNV(mFrameImage);
    }

    private Bitmap decodeNV(byte[] data) {
        int h = 720;
        int w = 1280;

        YuvImage yuvImage = new YuvImage(data, ImageFormat.NV21, w, h, null);
        Rect rect = new Rect(0, 0, w, h);
        ByteArrayOutputStream out_stream = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(rect, 100, out_stream);

        Bitmap image = BitmapFactory.decodeByteArray(out_stream.toByteArray(), 0, out_stream.size());
        return image;
    }

    public void saveImage(String name) {
        Bitmap bitmapImage = toBitmapImage();
        File directory = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DCIM);
        File filePath = new File(directory, String.format("s-%s.jpg", name));

        try (FileOutputStream fos = new FileOutputStream(filePath)) {
            bitmapImage.compress(Bitmap.CompressFormat.JPEG, 100, fos);
        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void toJson() {
        // TODO: 기록해서 brute force 최적화 돌려보기
    }

    @Override
    public String toString() {
        return "";
    }


    //
    // Optimization
    //
    public void setOptimalParam(double[] param) {
        mOptimalParam = param;
    }

    public double[] getOptimalParam() {
        return mOptimalParam;
    }

}
