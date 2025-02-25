package camp.visual.kappa.utils;

import android.content.Context;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public class FileUtils {
    public static final String FACE_TRACK_MODEL_NAME = "Face_Track.model";
    public static final String FACE_DETECT_MODEL_NAME = "Face_Detect.model";
    public static final String FACE_DETAIL_MODEL_NAME = "Face_Detail.model";
    public static final String EYEBALL_CONTOUR_MODEL_NAME = "Eyeball_Contour.model";

    public static void copyModelFiles(Context context) {
        copyFileIfNeed(context, FACE_TRACK_MODEL_NAME);
        copyFileIfNeed(context, FACE_DETECT_MODEL_NAME);
        copyFileIfNeed(context, FACE_DETAIL_MODEL_NAME);
        copyFileIfNeed(context, EYEBALL_CONTOUR_MODEL_NAME);
    }

    public static boolean copyFileIfNeed(Context context, String fileName) {
        String path = getFilePath(context, fileName);
        if (path != null) {
            File file = new File(path);
            if (!file.exists()) {
                try {
                    if (file.exists())
                        file.delete();

                    file.createNewFile();
                    InputStream in = context.getApplicationContext().getAssets().open(fileName);
                    if(in == null)
                    {
                        LogUtils.e("copyMode", "the src is not existed");
                        return false;
                    }
                    OutputStream out = new FileOutputStream(file);
                    byte[] buffer = new byte[4096];
                    int n;
                    while ((n = in.read(buffer)) > 0) {
                        out.write(buffer, 0, n);
                    }
                    in.close();
                    out.close();
                } catch (IOException e) {
                    file.delete();
                    return false;
                }
            }
        }
        return true;
    }

    public static String getFilePath(Context context, String fileName) {
        String path = null;
        File dataDir = context.getApplicationContext().getExternalFilesDir(null);
        if (dataDir != null) {
            path = dataDir.getAbsolutePath() + File.separator + fileName;
        }
        return path;
    }

}
