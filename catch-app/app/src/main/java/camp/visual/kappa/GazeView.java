package camp.visual.kappa;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.PointF;
import android.support.annotation.Nullable;
import android.util.AttributeSet;
import android.view.View;

public class GazeView extends View {
    private static final String TAG = "GazeView";

    private float mPointSize;
    private boolean onRender = false;
    private int screenWidth, screeHeight;

    private PointF mGazePoint;
    private Paint mGazePainter;
    private PointF mSecondPoint;
    private Paint mSecondPainter;
    private PointF mThirdPoint;
    private Paint mThirdPainter;


    public GazeView(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);

        mPointSize = 0.025f;
        mGazePoint = new PointF(0, 0);
        mSecondPoint = new PointF(0, 0);
        mThirdPoint = new PointF(0, 0);

        int gazeColor = Color.rgb(0, 250, 155);
        mGazePainter = new Paint(Paint.ANTI_ALIAS_FLAG);
        mGazePainter.setColor(gazeColor);

        int secColor = Color.rgb(255, 0, 255);
        mSecondPainter= new Paint(Paint.ANTI_ALIAS_FLAG);
        mSecondPainter.setColor(secColor);

        int thirdColor = Color.rgb(100, 0, 100);
        mThirdPainter = new Paint(Paint.ANTI_ALIAS_FLAG);
        mThirdPainter.setColor(thirdColor);
    }


    public void setRenderStatus(boolean onRender) {
        this.onRender = onRender;
        invalidate();
    }

    public void setGazePoint(float x, float y) {
        mGazePoint.set(x, y);
    }

    public void setSecondPoint(float x, float y) {
        mSecondPoint.set(x, y);
    }

    public void setThirdPoint(float x, float y) {
        mThirdPoint.set(x, y);
    }

    private void drawPoint(Canvas canvas, PointF point, Paint painter) {
        if (point == null) {
            return;
        }

        canvas.drawCircle(point.x, point.y, mPointSize * screenWidth, painter);
    }


    @Override
    protected void onDraw(Canvas canvas) {
        if (!onRender) {
            return;
        }

        drawPoint(canvas, mGazePoint, mGazePainter);
        drawPoint(canvas, mSecondPoint, mSecondPainter);
        drawPoint(canvas, mThirdPoint, mThirdPainter);
    }

    @Override
    protected void onLayout(boolean changed, int left, int top, int right, int bottom) {
        screenWidth = right - left;
        screeHeight = bottom - top;
    }

}
