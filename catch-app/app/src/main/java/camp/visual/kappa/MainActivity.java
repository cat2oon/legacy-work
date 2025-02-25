package camp.visual.kappa;

import android.Manifest;
import android.bluetooth.BluetoothClass;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.support.v4.app.Fragment;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.view.LayoutInflater;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.Toast;

import camp.visual.camera.CameraCalibrationActivity;
import camp.visual.device.Devices;
import camp.visual.kappa.cali.GazeActivity;
import camp.visual.kappa.utils.FileUtils;
import camp.visual.kappa.utils.STLicenseUtils;
import camp.visual.permission.Permissions;

public class MainActivity extends AppCompatActivity {

    private static final int PERMISSION_REQUEST_CAMERA = 0;
    private static final int PERMISSION_REQUEST_WRITE_EXTERNAL_STORAGE = 1;

    static {
        System.loadLibrary("opencv_java3");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Permissions.request(this);

        Toolbar toolbar = findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);

        checkLicense();
        FileUtils.copyModelFiles(this);

        Button trackingStartBtn = findViewById(R.id.btn_start_tracking);
        trackingStartBtn.setOnClickListener(v -> {
            if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                requestPermissions(new String[]{Manifest.permission.CAMERA}, PERMISSION_REQUEST_CAMERA);
            } else {
                startActivity(new Intent(getApplicationContext(), CameraActivity.class));
            }
        });

        Button gazeStartBtn = findViewById(R.id.btn_start_gaze);
        gazeStartBtn.setOnClickListener(v -> {
            startActivity(new Intent(getApplicationContext(), GazeActivity.class));
        });

        Button cameraCaliStartBtn = findViewById(R.id.btn_start_camera_calibration);
        cameraCaliStartBtn.setOnClickListener(v -> {
            startActivity(new Intent(getApplicationContext(), CameraCalibrationActivity.class));
        });

        setCameraParam();
    }

    private void checkLicense() {
        boolean mLicenseChecked = STLicenseUtils.checkLicense(this);
        String msg = mLicenseChecked ? "라이센스 성공" : "라이센스 실패";
        runOnUiThread(() -> Toast.makeText(getApplicationContext(), msg, Toast.LENGTH_SHORT).show());
    }

    private void setCameraParam() {
        Devices.setDevice(Devices.getGalaxyS8());
    }


    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        int id = item.getItemId();

        //noinspection SimplifiableIfStatement
        if (id == R.id.action_settings) {
            return true;
        }

        return super.onOptionsItemSelected(item);
    }

    public static class PlaceholderFragment extends Fragment {
        /**
         * The fragment argument representing the section number for this
         * fragment.
         */
        private static final String ARG_SECTION_NUMBER = "section_number";

        public PlaceholderFragment() {
        }

        /**
         * Returns a new instance of this fragment for the given section
         * number.
         */
        public static PlaceholderFragment newInstance(int sectionNumber) {
            PlaceholderFragment fragment = new PlaceholderFragment();
            Bundle args = new Bundle();
            args.putInt(ARG_SECTION_NUMBER, sectionNumber);
            fragment.setArguments(args);
            return fragment;
        }

        @Override
        public View onCreateView(LayoutInflater inflater, ViewGroup container,
                                 Bundle savedInstanceState) {
            return inflater.inflate(R.layout.fragment_main, container, false);
        }
    }

    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        if (requestCode == PERMISSION_REQUEST_CAMERA) {
            if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                startActivity(new Intent(getApplicationContext(), CameraActivity.class));
            } else {
                Toast.makeText(this, "카메라 권한 필요", Toast.LENGTH_SHORT) .show();
            }
        } else if(requestCode == PERMISSION_REQUEST_WRITE_EXTERNAL_STORAGE){
            if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                startActivity(new Intent(getApplicationContext(), CameraActivity.class));
            } else {
                Toast.makeText(this, "외부 저장 권한 필요", Toast.LENGTH_SHORT).show();
            }
        }
    }

}
