package camp.visual.ac;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.os.Environment;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;

public class Images {

    public static void saveImage(byte[] imageBytes, int width, int height, String filename) {
        // Bitmap bitmapImage = toBitmapImage(imageBytes, 1280, 720);
        Bitmap bitmapImage = toBitmapImage(imageBytes, width, height);
        File directory = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DCIM);
        File filePath = new File(directory, String.format("%s.jpg", filename));

        try (FileOutputStream fos = new FileOutputStream(filePath)) {
            bitmapImage.compress(Bitmap.CompressFormat.JPEG, 100, fos);
        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static Bitmap toBitmapImage(byte[] imageBytes, int width, int height) {
        YuvImage yuvImage = new YuvImage(imageBytes, ImageFormat.NV21, width, height, null);
        ByteArrayOutputStream out_stream = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new Rect(0, 0, width, height), 100, out_stream);

        return BitmapFactory.decodeByteArray(out_stream.toByteArray(), 0, out_stream.size());
    }

}
