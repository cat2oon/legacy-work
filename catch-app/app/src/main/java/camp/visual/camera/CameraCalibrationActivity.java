package camp.visual.camera;

import android.app.Activity;
import android.graphics.Color;
import android.graphics.PixelFormat;
import android.graphics.drawable.GradientDrawable;
import android.opengl.GLSurfaceView;
import android.os.Bundle;
import android.view.SurfaceView;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.TextView;

import camp.visual.kappa.R;


public class CameraCalibrationActivity extends Activity {
    private final static String TAG = "CameraCalibrationActivity";

    private Button mCaptureBtn;
    private TextView mLargerPreviewOption;      // 1270 x 1080
    private TextView mSmallerPreviewOption;     // 640 x 480
    private CameraCalibrationDisplay mCameraDisplay;


    /*
     * init
     */
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN, WindowManager.LayoutParams.FLAG_FULLSCREEN);
        setContentView(R.layout.activity_camera_calibration);

        initView();
        initCallback();
    }

    private void initView() {
        SurfaceView mSurfaceView = findViewById(R.id.surfaceView);
        GLSurfaceView glView = findViewById(R.id.id_gl_view);
        mCameraDisplay = new CameraCalibrationDisplay(getApplicationContext(), glView, mSurfaceView);
        mSurfaceView.getHolder().setFormat(PixelFormat.TRANSLUCENT);

        mCaptureBtn = findViewById(R.id.btn_capture);
        mLargerPreviewOption = findViewById(R.id.tv_1280_camera);
        mSmallerPreviewOption = findViewById(R.id.tv_640_camera);

        activatePreview(mLargerPreviewOption);
        deactivatePreview(mSmallerPreviewOption);
    }

    private void initCallback() {
        mSmallerPreviewOption.setOnClickListener(v -> {
            mCameraDisplay.changePreviewSize(1);
            activatePreview(mSmallerPreviewOption);
            deactivatePreview(mLargerPreviewOption);
        });

        mLargerPreviewOption.setOnClickListener(v -> {
            mCameraDisplay.changePreviewSize(0);
            activatePreview(mLargerPreviewOption);
            deactivatePreview(mSmallerPreviewOption);
        });

        mCaptureBtn.setOnClickListener(v -> {
            mCameraDisplay.capture();
        });
    }

    private void activatePreview(TextView view) {
        ((GradientDrawable) view.getBackground()).setColor(getResources().getColor(R.color.blue));
        view.setTextColor(getResources().getColor(R.color.white));
        view.setClickable(false);
    }

    private void deactivatePreview(TextView view) {
        ((GradientDrawable) view.getBackground()).setColor(Color.parseColor("#b2ffffff"));
        view.setTextColor(getResources().getColor(R.color.blue));
        view.setClickable(true);
    }



    /*
     * listeners
     */
    @Override
    protected void onResume() {
        super.onResume();
        mCameraDisplay.onResume();
    }

    @Override
    protected void onPause() {
        super.onPause();
        mCameraDisplay.onPause();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        mCameraDisplay.onDestroy();
    }

}

