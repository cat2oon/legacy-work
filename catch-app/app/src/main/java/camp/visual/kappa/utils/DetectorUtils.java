package camp.visual.kappa.utils;

import android.annotation.SuppressLint;
import android.content.Context;
import android.database.Cursor;
import android.database.sqlite.SQLiteException;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.PointF;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.net.Uri;
import android.provider.MediaStore;
import android.renderscript.Allocation;
import android.renderscript.Element;
import android.renderscript.RenderScript;
import android.renderscript.ScriptIntrinsicYuvToRGB;
import android.renderscript.Type;
import android.support.annotation.ColorInt;
import android.util.Log;
import android.util.TimingLogger;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;

public class DetectorUtils {
    private static final String TIMING_LOG_TAG = "DetectorUtils timing";

    private static Bitmap bitmap = null;
    private static Allocation ain = null;
    private static Allocation aOut = null;
    private static RenderScript mRS = null;
    private static ScriptIntrinsicYuvToRGB mYuvToRgb = null;


    public static int[] getBGRAImageByte(Bitmap image) {
        int width = image.getWidth();
        int height = image.getHeight();
        if (image.getConfig().equals(Config.ARGB_8888)) {
            int[] imgData = new int[width * height];
            image.getPixels(imgData, 0, width, 0, 0, width, height);
            return imgData;

        }

        return null;
    }

    public static byte[] getBGRFromBitmap(Bitmap bitmap) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        int componentsPerPixel = 3;
        int totalPixels = width * height;
        int totalBytes = totalPixels * componentsPerPixel;

        byte[] rgbValues = new byte[totalBytes];
        @ColorInt int[] argbPixels = new int[totalPixels];
        bitmap.getPixels(argbPixels, 0, width, 0, 0, width, height);
        for (int i = 0; i < totalPixels; i++) {
            @ColorInt int argbPixel = argbPixels[i];
            int red = Color.red(argbPixel);
            int green = Color.green(argbPixel);
            int blue = Color.blue(argbPixel);
            rgbValues[i * componentsPerPixel + 0] = (byte) blue;
            rgbValues[i * componentsPerPixel + 1] = (byte) green;
            rgbValues[i * componentsPerPixel + 2] = (byte) red;
        }

        return rgbValues;
    }

    @SuppressLint("NewApi")
    public static Bitmap NV21ToRGBABitmap(byte[] nv21, int width, int height, Context context) {
        TimingLogger timings = new TimingLogger(TIMING_LOG_TAG, "NV21ToRGBABitmap");
        Rect rect = new Rect(0, 0, width, height);

        try {
            Class.forName("android.renderscript.Element$DataKind").getField("PIXEL_YUV");
            Class.forName("android.renderscript.ScriptIntrinsicYuvToRGB");
            byte[] imageData = nv21;
            if (mRS == null) {
                mRS = RenderScript.create(context);
                mYuvToRgb = ScriptIntrinsicYuvToRGB.create(mRS, Element.U8_4(mRS));
                Type.Builder tb = new Type.Builder(mRS, Element.createPixel(mRS, Element.DataType.UNSIGNED_8, Element.DataKind.PIXEL_YUV));
                tb.setX(width);
                tb.setY(height);
                tb.setMipmaps(false);
                tb.setYuvFormat(ImageFormat.NV21);
                ain = Allocation.createTyped(mRS, tb.create(), Allocation.USAGE_SCRIPT);
                timings.addSplit("Prepare for ain");
                Type.Builder tb2 = new Type.Builder(mRS, Element.RGBA_8888(mRS));
                tb2.setX(width);
                tb2.setY(height);
                tb2.setMipmaps(false);
                aOut = Allocation.createTyped(mRS, tb2.create(), Allocation.USAGE_SCRIPT & Allocation.USAGE_SHARED);
                timings.addSplit("Prepare for aOut");
                bitmap = Bitmap.createBitmap(width, height, Config.ARGB_8888);
                timings.addSplit("Create Bitmap");
            }
            ain.copyFrom(imageData);
            timings.addSplit("ain copyFrom");
            mYuvToRgb.setInput(ain);
            timings.addSplit("setInput ain");
            mYuvToRgb.forEach(aOut);
            timings.addSplit("NV21 to ARGB forEach");
            aOut.copyTo(bitmap);
            timings.addSplit("Allocation to Bitmap");
        } catch (Exception e) {
            YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, width, height, null);
            timings.addSplit("NV21 bytes to YuvImage");

            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            yuvImage.compressToJpeg(rect, 90, baos);
            byte[] cur = baos.toByteArray();
            timings.addSplit("YuvImage crop and compress to Jpeg Bytes");

            bitmap = BitmapFactory.decodeByteArray(cur, 0, cur.length);
            timings.addSplit("Jpeg Bytes to Bitmap");
        }
        timings.dumpToLog();
        return bitmap;
    }

    static public Bitmap NV21ToRGBABitmap(byte[] nv21, int width, int height) {
        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, width, height, null);
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new Rect(0, 0, width, height), 100, baos);
        byte[] cur = baos.toByteArray();
        return BitmapFactory.decodeByteArray(cur, 0, cur.length);
    }


    static public void drawFaceRect(Canvas canvas, Rect rect, int width, int height) {
        if (canvas == null) {
            return;
        }

        Paint paint = new Paint();
        paint.setColor(Color.rgb(0, 218, 0));
        int strokeWidth = 3;
        if (canvas.getWidth() != 1080) {
            strokeWidth = (int) (strokeWidth * canvas.getWidth() / 1080);
        }
        paint.setStrokeWidth(strokeWidth);
        rect = adjustToScreenRectMin(rect, canvas.getWidth(), canvas.getHeight(), width, height);

        paint.setStyle(Style.STROKE);
        canvas.drawRect(rect, paint);
    }

    static public void drawPoints(Canvas canvas, Paint paint, PointF[] points, float[] visibles, int width, int height, int color) {
        if (canvas == null || points == null || points.length == 0) {
            return;
        }

        int strokeWidth = 2;
        if (width < 480) {
            strokeWidth = 1;
        }

        //眼球中心点
        if (points.length < 3) {
            strokeWidth = (int) (strokeWidth * 1.5);
        }

        for (int i = 0; i < points.length; i++) {
            PointF p = points[i];

            if (visibles != null && visibles[i] < 0.5) {
                paint.setColor(Color.rgb(255, 20, 20));
            } else {
                paint.setColor(color);
            }
            paint.setStyle(Style.FILL);
            canvas.drawCircle(p.x, p.y, strokeWidth, paint);
        }
        paint.setColor(color);
    }

    static public void drawFaceKeyPoints(Canvas canvas, Paint paint, PointF[] points, float[] visibles, int width, int height, int color) {
        if (canvas == null) return;

        int strokeWidth = 3;
        if (canvas.getWidth() != 1080) {
            strokeWidth = (int) (strokeWidth * canvas.getWidth() / 1080);
        }

        if (strokeWidth == 0) {
            strokeWidth = 1;
        }

        //眼球中心点
        if (points.length < 3) {
            strokeWidth = (int) (strokeWidth * 1.5);
        }

        //2D手关键点
        if (points.length == 20) {
            strokeWidth = strokeWidth * 2;
        }

        points = adjustToScreenPointsMin(points, canvas.getWidth(), canvas.getHeight(), width, height);

        for (int i = 0; i < points.length; i++) {
            PointF p = points[i];

            if (visibles != null && visibles[i] < 0.5) {
                paint.setColor(Color.rgb(255, 20, 20));
            } else {
                paint.setColor(color);
            }
            paint.setStyle(Style.FILL);
            canvas.drawCircle(p.x, p.y, strokeWidth, paint);
        }
        paint.setColor(color);
    }

    static public void drawPointsAndLines(Canvas canvas, Paint paint, PointF[] points, float[] visibles, int width, int height) {

        if (canvas == null) return;
        float value = 0.15f;

        if (points == null || points.length == 0) {
            return;
        }

        points = adjustToScreenPointsMin(points, canvas.getWidth(), canvas.getHeight(), width, height);

        int strokeWidth = 6;
        if (canvas.getWidth() >= 1080) {
            strokeWidth = (int) (strokeWidth * canvas.getWidth() / 1080);
        }
        paint.setStrokeWidth(strokeWidth);

        //画线
        paint.setColor(Color.parseColor("#0a8dff"));

        if (points.length == 14 && points.length == visibles.length) {
            ArrayList<STLine> bodyLines = new ArrayList<>();

            bodyLines.add(new STLine(points[0], points[1], visibles[0], visibles[1]));
            bodyLines.add(new STLine(points[2], points[4], visibles[2], visibles[4]));
            bodyLines.add(new STLine(points[4], points[6], visibles[4], visibles[6]));
            bodyLines.add(new STLine(points[3], points[5], visibles[3], visibles[5]));
            bodyLines.add(new STLine(points[5], points[7], visibles[5], visibles[7]));

            bodyLines.add(new STLine(points[8], points[9], visibles[8], visibles[9]));
            bodyLines.add(new STLine(points[8], points[10], visibles[8], visibles[10]));
            bodyLines.add(new STLine(points[10], points[12], visibles[10], visibles[12]));
            bodyLines.add(new STLine(points[9], points[11], visibles[9], visibles[11]));
            bodyLines.add(new STLine(points[11], points[13], visibles[11], visibles[13]));

            bodyLines.add(new STLine(points[2], points[3], visibles[2], visibles[3]));
            bodyLines.add(new STLine(points[2], points[8], visibles[2], visibles[8]));
            bodyLines.add(new STLine(points[3], points[9], visibles[3], visibles[9]));

            for (int i = 0; i < bodyLines.size(); i++) {
                if (bodyLines.get(i).startPointVisiable > value && bodyLines.get(i).endPointVisiable > value) {
                    canvas.drawLine(bodyLines.get(i).startPoint.x, bodyLines.get(i).startPoint.y, bodyLines.get(i).endPoint.x, bodyLines.get(i).endPoint.y, paint);
                }
            }

        } else if (points.length == 4 && points.length == visibles.length) {
            ArrayList<STLine> bodyLines = new ArrayList<>();

            bodyLines.add(new STLine(points[0], points[1], visibles[0], visibles[1]));
            bodyLines.add(new STLine(points[1], points[2], visibles[1], visibles[2]));
            bodyLines.add(new STLine(points[1], points[3], visibles[1], visibles[3]));

            for (int i = 0; i < bodyLines.size(); i++) {
                if (bodyLines.get(i).startPointVisiable > value && bodyLines.get(i).endPointVisiable > value) {
                    canvas.drawLine(bodyLines.get(i).startPoint.x, bodyLines.get(i).startPoint.y, bodyLines.get(i).endPoint.x, bodyLines.get(i).endPoint.y, paint);
                }
            }
        } else if (points.length >= 59 && points.length >= visibles.length) {
            ArrayList<STLine> bodyLines = new ArrayList<>();

            for (int i = 0; i < points.length - 1; i++) {
                bodyLines.add(new STLine(points[i], points[i + 1], visibles[i], visibles[i + 1]));
            }

            for (int i = 0; i < bodyLines.size(); i++) {
                if (bodyLines.get(i).startPointVisiable > value && bodyLines.get(i).endPointVisiable > value) {
                    canvas.drawLine(bodyLines.get(i).startPoint.x, bodyLines.get(i).startPoint.y, bodyLines.get(i).endPoint.x, bodyLines.get(i).endPoint.y, paint);
                }
            }
        }

        //画点
        for (int i = 0; i < points.length; i++) {
            PointF p = points[i];
            paint.setStyle(Style.FILL_AND_STROKE);

            if (visibles[i] > value) {
                paint.setColor(Color.rgb(0, 218, 0));
                canvas.drawCircle(p.x, p.y, strokeWidth, paint);
            }
        }

        paint.setColor(Color.rgb(0, 218, 0));
    }

    static public void drawPointsImage(Canvas canvas, Paint paint, PointF[] points, float[] visibles, int width, int height, boolean backCamera) {

        if (canvas == null) return;
        int strokeWidth = 2;

        if (width > 2048) {
            strokeWidth = 3;
        }

        for (int i = 0; i < points.length; i++) {
            PointF p = points[i];
            if (backCamera) {
                p.y = width - p.y;
            }
            if (visibles[i] < 0.5) {
                paint.setColor(Color.rgb(255, 20, 20));
            } else {
                paint.setColor(Color.rgb(0, 255, 0));
            }
            canvas.drawCircle(p.x, p.y, strokeWidth, paint);
        }
        paint.setColor(Color.rgb(0, 255, 0));
    }

    static public void drawPoints(Canvas canvas, Paint paint, PointF[] points, int width, int height, boolean backCamera) {

        if (canvas == null) return;
        int strokeWidth = Math.max(width / 360, 2);

        for (int i = 0; i < points.length; i++) {
            PointF p = points[i];
            if (backCamera) {
                p.y = width - p.y;
            }
            paint.setColor(Color.rgb(0, 0, 255));
            canvas.drawCircle(p.x, p.y, strokeWidth, paint);
        }
    }

    static public Rect RotateDeg90(Rect rect, int width, int height) {
        int left = rect.left;
        rect.left = height - rect.bottom;
        rect.bottom = rect.right;
        rect.right = height - rect.top;
        rect.top = left;
        return rect;
    }

    static public Rect RotateDeg270(Rect rect, int width, int height) {
        int left = rect.left;
        rect.left = rect.top;
        rect.top = width - rect.right;
        rect.right = rect.bottom;
        rect.bottom = width - left;
        return rect;
    }

    static public Rect RotateDeg270AndMirrow(Rect rect, int width, int height) {
        int left = rect.left;
        rect.left = rect.top;
        rect.top = width - rect.right;
        rect.right = rect.bottom;
        rect.bottom = width - left;

        Rect rectNew = new Rect();
        rectNew.left = height - rect.right;
        rectNew.right = height - rect.left;
        rectNew.top = rect.top;
        rectNew.bottom = rect.bottom;
        return rectNew;
    }

    static public Rect RotateDeg90AndMirrow(Rect rect, int width, int height) {
        int left = rect.left;
        rect.left = height - rect.bottom;
        rect.bottom = rect.right;
        rect.right = height - rect.top;
        rect.top = left;

        Rect rectNew = new Rect();
        rectNew.left = height - rect.right;
        rectNew.right = height - rect.left;
        rectNew.top = rect.top;
        rectNew.bottom = rect.bottom;
        return rectNew;
    }

    static public PointF RotateDeg90(PointF point, int width, int height) {
        float x = point.x;
        point.x = height - point.y;
        point.y = x;
        return point;
    }

    static public PointF RotateDeg90AndMirrow(PointF point, int width, int height) {
        float x = point.x;
        point.x = height - point.y;
        point.y = x;

        point.x = height - point.x;
        return point;
    }

    static public PointF RotateDeg270AndMirrow(PointF point, int width, int height) {
        float x = point.x;
        point.x = point.y;
        point.y = width - x;

        point.x = height - point.x;
        return point;
    }

    static public PointF RotateDeg270(PointF point, int width, int height) {
        float x = point.x;
        point.x = point.y;
        point.y = width - x;
        return point;
    }

    public static Bitmap getRotateBitmap(Bitmap bitmap, int rotation) {
        if (null == bitmap || bitmap.isRecycled()) {
            return null;
        }

        Matrix matrix = new Matrix();
        matrix.postRotate(rotation);
        Bitmap cropBitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, false);
        return cropBitmap;
    }

    public static void recycleBitmap(Bitmap bitmap) {
        if (bitmap == null || bitmap.isRecycled()) {
            return;
        }
        bitmap.recycle();
        bitmap = null;
    }

    public static void copyModelIfNeed(String modelName, Context mContext) {
        String path = getModelPath(modelName, mContext);
        if (path != null) {
            File modelFile = new File(path);
            if (!modelFile.exists()) {
                //如果模型文件不存在或者当前模型文件的版本跟sdcard中的版本不一样
                try {
                    if (modelFile.exists())
                        modelFile.delete();
                    modelFile.createNewFile();
                    InputStream in = mContext.getApplicationContext().getAssets().open(modelName);
                    if (in == null) {
                        Log.e("MultiTrack106", "the src module is not existed");
                    }
                    OutputStream out = new FileOutputStream(modelFile);
                    byte[] buffer = new byte[4096];
                    int n;
                    while ((n = in.read(buffer)) > 0) {
                        out.write(buffer, 0, n);
                    }
                    in.close();
                    out.close();
                } catch (IOException e) {
                    modelFile.delete();
                }
            }
        }
    }

    public static String getModelPath(String modelName, Context mContext) {
        String path = null;
        File dataDir = mContext.getApplicationContext().getExternalFilesDir(null);
        if (dataDir != null) {
            path = dataDir.getAbsolutePath() + File.separator + modelName;
        }
        return path;
    }

    public static Bitmap getBitmapFromFile(Uri uri) {
        if (uri == null) {
            return null;
        }

        Bitmap bmp = null;
        BitmapFactory.Options opts = new BitmapFactory.Options();
        opts.inJustDecodeBounds = true;
        bmp = BitmapFactory.decodeFile(uri.getPath(), opts);
        opts.inSampleSize = computeSampleSize(opts);
        opts.inJustDecodeBounds = false;
        bmp = BitmapFactory.decodeFile(uri.getPath(), opts);

        return bmp;
    }

    public static Bitmap getBitmapAfterRotate(Uri uri, Context context) {
        Bitmap rotatebitmap = null;
        Bitmap srcbitmap = null;
        String[] filePathColumn = {MediaStore.Images.Media.DATA, MediaStore.Images.Media.ORIENTATION};
        Cursor cursor = null;
        String picturePath = null;
        String orientation = null;

        try {
            cursor = context.getContentResolver().query(uri, filePathColumn, null, null, null);

            if (cursor != null) {
                cursor.moveToFirst();
                int columnIndex = cursor.getColumnIndex(filePathColumn[0]);
                picturePath = cursor.getString(columnIndex);
                orientation = cursor.getString(cursor.getColumnIndex(filePathColumn[1]));
            }
        } catch (SQLiteException e) {
            // Do nothing
        } catch (IllegalArgumentException e) {
            // Do nothing
        } catch (IllegalStateException e) {
            // Do nothing
        } finally {
            if (cursor != null)
                cursor.close();
        }
        if (picturePath != null) {
            int angle = 0;
            if (orientation != null && !"".equals(orientation)) {
                angle = Integer.parseInt(orientation);
            }

            BitmapFactory.Options opts = new BitmapFactory.Options();
            opts.inJustDecodeBounds = true;
            srcbitmap = BitmapFactory.decodeFile(picturePath, opts);

            opts.inSampleSize = computeSampleSize(opts);
            opts.inJustDecodeBounds = false;
            srcbitmap = BitmapFactory.decodeFile(picturePath, opts);
            if (angle != 0) {
                // 下面的方法主要作用是把图片转一个角度，也可以放大缩小等
                Matrix m = new Matrix();
                int width = srcbitmap.getWidth();
                int height = srcbitmap.getHeight();
                m.setRotate(angle); // 旋转angle度
                try {
                    rotatebitmap = Bitmap.createBitmap(srcbitmap, 0, 0, width, height, m, true);// 新生成图片
                } catch (Exception e) {

                }

                if (rotatebitmap == null) {
                    rotatebitmap = srcbitmap;
                }

                if (srcbitmap != rotatebitmap) {
                    srcbitmap.recycle();
                }
            } else {
                rotatebitmap = srcbitmap;
            }
        }

        return rotatebitmap;
    }

    public static int computeSampleSize(BitmapFactory.Options opts) {
        int sampleSize = 1;
        int width = opts.outWidth;
        int height = opts.outHeight;
        if (width > 2048 || height > 2048) {
            sampleSize = 4;
        } else if (width > 1024 || height > 1024) {
            sampleSize = 2;
        }

        return sampleSize;
    }

    public static Rect adjustToScreenRectMax(Rect intputRect, int screenWidth, int screenHeight, int imageWidth, int imageHeight) {
        Rect rect = intputRect;

        if (intputRect == null) {
            return null;
        }

        if ((float) screenHeight / screenWidth >= (float) imageHeight / imageWidth) {
            float rate = ((float) screenWidth / imageWidth * imageHeight) / screenHeight;
            rect.top = (int) ((float) imageHeight / 2 - ((float) imageHeight / 2 - rect.top) * rate);
            rect.bottom = (int) ((float) imageHeight / 2 - ((float) imageHeight / 2 - rect.bottom) * rate);
            rect.left = intputRect.left;
            rect.right = intputRect.right;
        } else {
            rect.top = intputRect.top;
            rect.bottom = intputRect.bottom;
            float rate = ((float) screenHeight / imageHeight * imageWidth) / screenWidth;
            rect.left = (int) ((float) imageWidth / 2 - ((float) imageWidth / 2 - rect.left) * rate);
            rect.right = (int) ((float) imageWidth / 2 - ((float) imageWidth / 2 - rect.right) * rate);
        }

        return rect;
    }

    public static PointF[] adjustToScreenPointsMax(PointF[] points, int screenWidth, int screenHeight, int imageWidth, int imageHeight) {
        PointF[] intputPoints = points;

        if (points == null || points.length == 0) {
            return null;
        }

        if ((float) screenHeight / screenWidth >= (float) imageHeight / imageWidth) {
            float rate = ((float) screenWidth / imageWidth * imageHeight) / screenHeight;
            for (int i = 0; i < intputPoints.length; i++) {
                float x = intputPoints[i].x;
                float y = (float) imageHeight / 2 - ((float) imageHeight / 2 - intputPoints[i].y) * rate;
                intputPoints[i].set(x, y);
            }
        } else {
            float rate = ((float) screenHeight / imageHeight * imageWidth) / screenWidth;
            for (int i = 0; i < intputPoints.length; i++) {
                float x = (float) imageWidth / 2 - ((float) imageWidth / 2 - intputPoints[i].x) * rate;
                float y = intputPoints[i].y;

                intputPoints[i].set(x, y);
            }
        }

        return intputPoints;
    }

    public static Rect adjustToScreenRectMin(Rect intputRect, int screenWidth, int screenHeight, int imageWidth, int imageHeight) {
        Rect rect = intputRect;

        if (intputRect == null) {
            return null;
        }

        if ((float) screenHeight / screenWidth >= (float) imageHeight / imageWidth) {
            float rate = (float) screenHeight / imageHeight;
            rect.top = (int) (intputRect.top * rate);
            rect.bottom = (int) (intputRect.bottom * rate);
            rect.left = (int) ((float) screenWidth / 2 - ((float) imageWidth / 2 - rect.left) * rate);
            rect.right = (int) ((float) screenWidth / 2 - ((float) imageWidth / 2 - rect.right) * rate);
        } else {
            float rate = (float) screenWidth / imageWidth;
            rect.top = (int) ((float) screenHeight / 2 - ((float) imageHeight / 2 - rect.top) * rate);
            rect.bottom = (int) ((float) screenHeight / 2 - ((float) imageHeight / 2 - rect.bottom) * rate);
            rect.left = (int) (intputRect.left * rate);
            rect.right = (int) (intputRect.right * rate);
        }

        return rect;
    }

    public static PointF[] adjustToScreenPointsMin(PointF[] points, int screenWidth, int screenHeight, int imageWidth, int imageHeight) {
        PointF[] intputPoints = points;

        if (points == null || points.length == 0) {
            return null;
        }

        if ((float) screenHeight / screenWidth >= (float) imageHeight / imageWidth) {
            float rate = (float) screenHeight / imageHeight;
            for (int i = 0; i < intputPoints.length; i++) {
                float x = (float) screenWidth / 2 - ((float) imageWidth / 2 - intputPoints[i].x) * rate;
                float y = intputPoints[i].y * rate;
                intputPoints[i].set(x, y);
            }
        } else {
            float rate = (float) screenWidth / imageWidth;
            for (int i = 0; i < intputPoints.length; i++) {
                float x = intputPoints[i].x * rate;
                float y = (float) screenHeight / 2 - ((float) imageHeight / 2 - intputPoints[i].y) * rate;
                intputPoints[i].set(x, y);
            }
        }
        return intputPoints;
    }

}
