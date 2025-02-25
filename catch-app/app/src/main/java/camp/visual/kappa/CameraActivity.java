package camp.visual.kappa;

import android.app.Activity;
import android.graphics.Color;
import android.graphics.PixelFormat;
import android.graphics.drawable.GradientDrawable;
import android.opengl.GLSurfaceView;
import android.os.Bundle;
import android.util.Pair;
import android.view.SurfaceView;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.RelativeLayout;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;

import org.eclipse.collections.impl.map.mutable.UnifiedMap;

import camp.visual.device.BaseDevice;
import camp.visual.device.Devices;
import camp.visual.kappa.display.CameraDisplay;
import camp.visual.kappa.utils.Accelerometer;
import camp.visual.kappa.utils.LogUtils;
import camp.visual.kappa.utils.STLicenseUtils;

import static java.lang.Thread.sleep;


public class CameraActivity extends Activity implements View.OnClickListener {
    private final static String TAG = "CameraActivity";

    private BaseDevice mDevice = Devices.getDevice();
    private Thread mCpuInfoThread;
    private CameraDisplay mCameraDisplay;

    private boolean mOnSnapshot;
    private GazeView mGazeView;
    private Button mBtnSnapshot;
    private TextView mLargerPreviewSizeOption;      // 1270 x 1080
    private TextView mSmallerPreviewSizeOption;     // 640 x 480
    private TextView mInfoViewOption;
    private TextView mRenderViewOption;
    private TextView mRenderLandmarkOption;

    private Switch mFaceBaseSwitch;
    private Switch mFaceDetailSwitch;
    private Switch mEyeBallCenterSwitch;
    private Switch mEyeBallContourSwitch;

    // View
    private SurfaceView mSurfaceViewOverlap;
    private GLSurfaceView mGlSurfaceView;


    /*
     * init
     */
    private void initView() {
        new Accelerometer(getApplicationContext());     // TODO: createInstance singleton

        mGlSurfaceView = findViewById(R.id.id_gl_sv);
        mSurfaceViewOverlap = findViewById(R.id.surfaceViewOverlap);
        mCameraDisplay = new CameraDisplay(getApplicationContext(), mGlSurfaceView, mSurfaceViewOverlap);

        mBtnSnapshot = findViewById(R.id.btn_snapshot);
        mGazeView = findViewById(R.id.gazeView);
        mSurfaceViewOverlap.getHolder().setFormat(PixelFormat.TRANSLUCENT);

        // Upper tab
        mLargerPreviewSizeOption = findViewById(R.id.tv_larger_preview_size);
        mSmallerPreviewSizeOption = findViewById(R.id.tv_smaller_preview_size);
        mRenderLandmarkOption = findViewById(R.id.tv_show_render);
        mRenderViewOption = findViewById(R.id.tv_view_render);
        mInfoViewOption = findViewById(R.id.tv_info_render);

        TextView mSettingDone = findViewById(R.id.tv_setting_done);
        mSettingDone.setOnClickListener(v -> {
            findViewById(R.id.rv_setting_bg).setVisibility(View.GONE);
            findViewById(R.id.ll_select_options).setVisibility(View.VISIBLE);
        });

        RelativeLayout mSettingBackground = findViewById(R.id.rv_setting_bg);
        mSettingBackground.setOnClickListener(v -> {
            findViewById(R.id.rv_setting_bg).setVisibility(View.GONE);
            findViewById(R.id.ll_select_options).setVisibility(View.VISIBLE);
        });

        ImageView mSetting = findViewById(R.id.iv_setting_options_switch);
        mSetting.setOnClickListener(v -> {
            findViewById(R.id.ll_select_options).setVisibility(View.GONE);
            findViewById(R.id.rv_setting_bg).setVisibility(View.VISIBLE);
        });
    }

    private void initEvents() {
        if (!STLicenseUtils.checkLicense(CameraActivity.this)) {
            String msg = "인증 필요함!";
            runOnUiThread(() -> Toast.makeText(getApplicationContext(), msg, Toast.LENGTH_SHORT).show());
            finish();
            return;
        }

        initPreview();
        initCallback();
        initSwitches();
        initActivateSwitch();
    }

    private void initSwitches() {
        Switch mShowRenderSwitch = findViewById(R.id.sw_show_render_switch);
        mShowRenderSwitch.setOnCheckedChangeListener((buttonView, isChecked) -> {
            mCameraDisplay.setShowRender(isChecked);
        });

        mFaceBaseSwitch = findViewById(R.id.sw_face106_switch);
        mFaceBaseSwitch.setOnCheckedChangeListener((buttonView, isChecked) -> {
            if (isChecked) {
                mCameraDisplay.setEnableFace106(true);
                findViewById(R.id.ll_face_info).setVisibility(View.VISIBLE);
            } else {
                mFaceDetailSwitch.setChecked(false);
                mEyeBallCenterSwitch.setChecked(false);
                mEyeBallContourSwitch.setChecked(false);

                mCameraDisplay.setEnableFace106(false);
                mCameraDisplay.setEnableFaceExtra(false);
                mCameraDisplay.setEnableEyeBallCenter(false);
                mCameraDisplay.setEnableEyeBallContour(false);

                findViewById(R.id.ll_face_info).setVisibility(View.INVISIBLE);
            }
        });

        mFaceDetailSwitch = findViewById(R.id.sw_face_extra_switch);
        mFaceDetailSwitch.setOnCheckedChangeListener((buttonView, isChecked) -> {
            if (isChecked) {
                mFaceBaseSwitch.setChecked(true);
                mCameraDisplay.setEnableFace106(true);
                mCameraDisplay.setEnableFaceExtra(true);
                findViewById(R.id.ll_face_info).setVisibility(View.VISIBLE);
            } else {
                mCameraDisplay.setEnableFaceExtra(false);
            }
        });

        mEyeBallCenterSwitch = findViewById(R.id.sw_eyeball_center_switch);
        mEyeBallCenterSwitch.setOnCheckedChangeListener((buttonView, isChecked) -> {
            if (isChecked) {
                mFaceBaseSwitch.setChecked(true);
                mCameraDisplay.setEnableFace106(true);
                mCameraDisplay.setEnableEyeBallCenter(true);
                findViewById(R.id.ll_face_info).setVisibility(View.VISIBLE);
            } else {
                mCameraDisplay.setEnableEyeBallCenter(false);
            }
        });

        mEyeBallContourSwitch = findViewById(R.id.sw_eyeball_contour_switch);
        mEyeBallContourSwitch.setOnCheckedChangeListener((buttonView, isChecked) -> {
            if (isChecked) {
                mFaceBaseSwitch.setChecked(true);
                mCameraDisplay.setEnableFace106(true);
                mCameraDisplay.setEnableEyeBallContour(true);
                findViewById(R.id.ll_face_info).setVisibility(View.VISIBLE);
            } else {
                mCameraDisplay.setEnableEyeBallContour(false);
            }
        });
    }

    private void initActivateSwitch() {
        mFaceBaseSwitch.setChecked(true);
        mCameraDisplay.setEnableFace106(true);
    }

    private void initPreview() {
        ((GradientDrawable) mSmallerPreviewSizeOption.getBackground()).setColor(Color.parseColor("#b2ffffff"));
        ((GradientDrawable) mLargerPreviewSizeOption.getBackground()).setColor(getResources().getColor(R.color.blue));
        mLargerPreviewSizeOption.setTextColor(getResources().getColor(R.color.white));
        mLargerPreviewSizeOption.setClickable(false);

        mSmallerPreviewSizeOption.setOnClickListener(v -> {
            if (mCameraDisplay != null && !mCameraDisplay.isChangingPreviewSize()) {
                mCameraDisplay.changePreviewSize(1);

                ((GradientDrawable) mLargerPreviewSizeOption.getBackground()).setColor(Color.parseColor("#b2ffffff"));
                ((GradientDrawable) mSmallerPreviewSizeOption.getBackground()).setColor(getResources().getColor(R.color.blue));
                mSmallerPreviewSizeOption.setTextColor(getResources().getColor(R.color.white));
                mSmallerPreviewSizeOption.setClickable(false);
                mLargerPreviewSizeOption.setTextColor(getResources().getColor(R.color.blue));
                mLargerPreviewSizeOption.setClickable(true);
            }
        });

        mLargerPreviewSizeOption.setOnClickListener(v -> {
            if (mCameraDisplay != null && !mCameraDisplay.isChangingPreviewSize()) {
                mCameraDisplay.changePreviewSize(0);

                ((GradientDrawable) mLargerPreviewSizeOption.getBackground()).setColor(getResources().getColor(R.color.blue));
                ((GradientDrawable) mSmallerPreviewSizeOption.getBackground()).setColor(Color.parseColor("#b2ffffff"));
                mLargerPreviewSizeOption.setTextColor(getResources().getColor(R.color.white));
                mLargerPreviewSizeOption.setClickable(false);
                mSmallerPreviewSizeOption.setTextColor(getResources().getColor(R.color.blue));
                mSmallerPreviewSizeOption.setClickable(true);
            }
        });

        mRenderLandmarkOption.setOnClickListener(v -> {
            if (mCameraDisplay != null && !mCameraDisplay.isChangingPreviewSize()) {
                mCameraDisplay.toggleRender();
            }
        });

        mRenderViewOption.setOnClickListener(v -> {
            if (mCameraDisplay != null && !mCameraDisplay.isChangingPreviewSize()) {
                toggleView();
            }
        });

        View cpuCost = findViewById(R.id.ll_frame_cost);
        View faceInfo = findViewById(R.id.ll_face_info);
        mInfoViewOption.setOnClickListener(v -> {
            int visibility;
            if (cpuCost.getVisibility() == View.VISIBLE) {
                visibility = View.INVISIBLE;
            } else {
                visibility = View.VISIBLE;
            }

            cpuCost.setVisibility(visibility);
            faceInfo.setVisibility(visibility);
        });
    }

    private void initCallback() {
         mBtnSnapshot.setOnClickListener(v -> {
            if (mOnSnapshot) {
                mCameraDisplay.requestSnapshot();
            } else {
                mCameraDisplay.requestCalibration();
                mBtnSnapshot.setVisibility(View.INVISIBLE);
            }
        });

         mCameraDisplay.setUI(this);
         updateUI();
    }


    /*
     * Calibration & Snapshot
     */
    public void updateUI() {
        int numSnapshot = mCameraDisplay.getNumSnapshot();
        int numScenario = mCameraDisplay.getNumScenario();
        mOnSnapshot = numSnapshot < numScenario;
        String btnText = String.valueOf(numSnapshot+1);

        if (mOnSnapshot) {
            Pair<Double, Double> c = mCameraDisplay.getCurrentTarget();
            drawTarget(c.first, c.second);
        } else {
            drawTarget(0, 0);
            btnText = "Calibration";
        }

        endDraw();
        mBtnSnapshot.setText(btnText);
    }

    public void drawTarget(double mx, double my) {
        double[] pxy = mDevice.pixelFromMillisByLens(mx, my);
        mGazeView.setGazePoint((int)pxy[0], (int)pxy[1]);
    }

    public void drawSecondTarget(double mx, double my) {
        double[] pxy = mDevice.pixelFromMillisByLens(mx, my);
        mGazeView.setSecondPoint((int)pxy[0], (int)pxy[1]);
    }

    public void drawThirdTarget(double mx, double my) {
        double[] pxy = mDevice.pixelFromMillisByLens(mx, my);
        mGazeView.setThirdPoint((int)pxy[0], (int)pxy[1]);
    }

    public void endDraw() {
        mGazeView.setRenderStatus(true);
    }


    /*
     * View
     */
    public void toggleView() {
        toggleCameraView();
        mCameraDisplay.setShowRender(false);
    }

    public void toggleCameraView() {
        if (mSurfaceViewOverlap.getVisibility() == View.VISIBLE) {
            mGlSurfaceView.setVisibility(View.INVISIBLE);
            mSurfaceViewOverlap.setVisibility(View.INVISIBLE);

        } else {
            mGlSurfaceView.setVisibility(View.VISIBLE);
            mSurfaceViewOverlap.setVisibility(View.VISIBLE);
        }
    }


    /*
     * miscellaneous
     */
    private void renderCpuInfo() {
        if (mCameraDisplay == null) {
            return;
        }

        TextView tvFps = findViewById(R.id.tv_fps);
        TextView tvFrameCost = findViewById(R.id.tv_frame_cost);
        tvFps.setText(String.format("Fps\n%d", mCameraDisplay.getFps()));
        // tvFrameCost.setText(String.format("frameCost\n%d", mCameraDisplay.getFrameCost()));

        showFaceInfo(mCameraDisplay.getFaceInfo());
    }

    private void showFaceInfo(UnifiedMap faceInfo) {
        if (faceInfo == null || faceInfo.isEmpty()) {
            return;
        }

        String ER = (String) faceInfo.get("ER");
        if (ER != null) {
            ((TextView) findViewById(R.id.tv_ER)).setText(String.format("ER\n%s", ER));
        }

        /*
        String IR = (String) faceInfo.get("IR");
        if (IR != null) {
            ((TextView) findViewById(R.id.tv_EP)).setText(String.format("IR\n%s", IR));
        }
        */

        String EP = (String) faceInfo.get("EP");
        if (EP != null) {
            ((TextView) findViewById(R.id.tv_EP)).setText(String.format("EP\n%s", EP));
        }

        String PR = (String) faceInfo.get("PR");
        if (PR != null) {
            ((TextView) findViewById(R.id.tv_PR)).setText(String.format("PR\n%s", PR));
        }

        /*
        String YPR = (String) faceInfo.get("YPR");
        if (YPR != null) {
            ((TextView) findViewById(R.id.tv_IR)).setText(String.format("YPR\n%s", YPR));
        }
        */

    }

    private void startShowCpuInfo() {
        mCpuInfoThread = new Thread(() -> {
            while (true) {
                runOnUiThread(this::renderCpuInfo);
                try {
                    sleep(100);
                }
                catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        });
        mCpuInfoThread.start();
    }

    private void stopShowCpuInfo() {
        if (mCpuInfoThread == null) {
            return;
        }

        mCpuInfoThread.interrupt();
        mCpuInfoThread = null;
    }


    /*
     * listeners
     */
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN, WindowManager.LayoutParams.FLAG_FULLSCREEN);
        setContentView(R.layout.activity_surface_view);

        initView();
        initEvents();

        if (mCameraDisplay.mCalibrationSuccess.get()) {
            mBtnSnapshot.setVisibility(View.INVISIBLE);
        }
    }

    @Override
    public void onClick(View v) {
        switch (v.getId()) {
            default:
                break;
        }
    }

    @Override
    protected void onResume() {
        LogUtils.i(TAG, "onResume");
        super.onResume();
        mCameraDisplay.onResume();
        startShowCpuInfo();
    }

    @Override
    protected void onPause() {
        LogUtils.i(TAG, "onPause");
        super.onPause();
        mCameraDisplay.onPause();
        stopShowCpuInfo();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        mCameraDisplay.onDestroy();
    }

}

