package camp.visual.kappa.cali;

import android.app.Activity;
import android.graphics.Color;
import android.graphics.PixelFormat;
import android.graphics.drawable.GradientDrawable;
import android.opengl.GLSurfaceView;
import android.os.Bundle;
import android.view.SurfaceView;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.ImageView;
import android.widget.RelativeLayout;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;

import org.eclipse.collections.impl.map.mutable.UnifiedMap;

import camp.visual.device.BaseDevice;
import camp.visual.device.Devices;
import camp.visual.kappa.GazeView;
import camp.visual.kappa.R;
import camp.visual.kappa.utils.Accelerometer;
import camp.visual.kappa.utils.LogUtils;
import camp.visual.kappa.utils.STLicenseUtils;

import static java.lang.Thread.sleep;



public class GazeActivity extends Activity  {
    private static String TAG = "GazeActivity";

    private Thread mCpuInfoThread;
    private GazeDisplay mGazeDisplay;
    private BaseDevice mDevice = Devices.getDevice();

    private GazeView mGazeView;
    private TextView mLagerPreviewOption;      // 1270 x 1080
    private TextView mSmallerPreviewOption;    // 640 x 480
    private TextView mInfoViewOption;
    private TextView mRenderViewOption;
    private TextView mRenderLandmarkOption;

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
        mGazeDisplay = new GazeDisplay(getApplicationContext(), mGlSurfaceView, mSurfaceViewOverlap);

        mGazeView = findViewById(R.id.gazeView);
        mSurfaceViewOverlap.getHolder().setFormat(PixelFormat.TRANSLUCENT);

        // Upper tab
        mLagerPreviewOption = findViewById(R.id.tv_larger_preview_size);
        mSmallerPreviewOption = findViewById(R.id.tv_smaller_preview_size);
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
        if (!STLicenseUtils.checkLicense(GazeActivity.this)) {
            String msg = "인증 필요함!";
            runOnUiThread(() -> Toast.makeText(getApplicationContext(), msg, Toast.LENGTH_SHORT).show());
            finish();
            return;
        }

        initPreview();
        initCallback();
        initSwitches();
    }

    private void initSwitches() {
        Switch mShowRenderSwitch = findViewById(R.id.sw_show_render_switch);
        mShowRenderSwitch.setOnCheckedChangeListener((buttonView, isChecked) -> {
            mGazeDisplay.setShowRender(isChecked);
        });
    }

    private void initPreview() {
        activatePreview(mLagerPreviewOption);
        deactivatePreview(mSmallerPreviewOption);

        mSmallerPreviewOption.setOnClickListener(v -> {
            if (mGazeDisplay != null && !mGazeDisplay.isChangingPreviewSize()) {
                mGazeDisplay.changePreviewSize(1);
                activatePreview(mSmallerPreviewOption);
                deactivatePreview(mLagerPreviewOption);
            }
        });

        mLagerPreviewOption.setOnClickListener(v -> {
            if (mGazeDisplay != null && !mGazeDisplay.isChangingPreviewSize()) {
                mGazeDisplay.changePreviewSize(0);
                activatePreview(mLagerPreviewOption);
                deactivatePreview(mSmallerPreviewOption);
            }
        });

        mRenderLandmarkOption.setOnClickListener(v -> {
            if (mGazeDisplay != null && !mGazeDisplay.isChangingPreviewSize()) {
                mGazeDisplay.toggleRender();
            }
        });

        mRenderViewOption.setOnClickListener(v -> {
            if (mGazeDisplay != null && !mGazeDisplay.isChangingPreviewSize()) {
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

    private void initCallback() {
         mGazeDisplay.setUI(this);
    }


    /*
     * View
     */
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

    public void toggleView() {
        toggleCameraView();
        mGazeDisplay.setShowRender(false);
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
        if (mGazeDisplay == null) {
            return;
        }

        TextView tvFps = findViewById(R.id.tv_fps);
        tvFps.setText(String.format("Fps\n%d", mGazeDisplay.getFps()));
        // TextView tvFrameCost = findViewById(R.id.tv_frame_cost);
        // tvFrameCost.setText(String.format("frameCost\n%d", mGazeDisplay.getFrameCost()));

        showFaceInfo(mGazeDisplay.getFaceInfo());
    }

    private void showFaceInfo(UnifiedMap faceInfo) {
        if (faceInfo == null || faceInfo.isEmpty()) {
            return;
        }

        String ER = (String) faceInfo.get("ER");
        if (ER != null) {
            ((TextView) findViewById(R.id.tv_ER)).setText(String.format("ER\n%s", ER));
        }

        String IR = (String) faceInfo.get("IR");
        if (IR != null) {
            ((TextView) findViewById(R.id.tv_IR)).setText(String.format("IR\n%s", IR));
        }

        String EP = (String) faceInfo.get("EP");
        if (EP != null) {
            ((TextView) findViewById(R.id.tv_EP)).setText(String.format("EP\n%s", EP));
        }

        String PR = (String) faceInfo.get("PR");
        if (PR != null) {
            ((TextView) findViewById(R.id.tv_PR)).setText(String.format("PR\n%s", PR));
        }
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
    }

    @Override
    protected void onResume() {
        LogUtils.i(TAG, "onResume");
        super.onResume();
        mGazeDisplay.onResume();
        startShowCpuInfo();
    }

    @Override
    protected void onPause() {
        LogUtils.i(TAG, "onPause");
        super.onPause();
        mGazeDisplay.onPause();
        stopShowCpuInfo();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        mGazeDisplay.onDestroy();
    }

}

