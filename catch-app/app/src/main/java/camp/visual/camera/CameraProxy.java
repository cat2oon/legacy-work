package camp.visual.camera;

import android.content.Context;
import android.graphics.ImageFormat;
import android.graphics.Point;
import android.graphics.SurfaceTexture;
import android.hardware.Camera;
import android.hardware.Camera.CameraInfo;
import android.hardware.Camera.Parameters;
import android.hardware.Camera.PreviewCallback;
import android.hardware.Camera.Size;
import android.util.Log;
import android.view.SurfaceHolder;

import org.apache.commons.lang3.exception.ExceptionUtils;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class CameraProxy {
	private static final String TAG = "CameraProxy";

	private int mCameraId;
	private Camera mCamera;
	private CameraInfo mCameraInfo = new CameraInfo();


    public CameraProxy(Context context) { }


	/*
	 * APIs
	 */
	public boolean openCamera(int cameraId) {
		try {
			releaseCamera();
			mCamera = Camera.open(cameraId);
			mCamera.getParameters();
			mCameraId = cameraId;
			Camera.getCameraInfo(cameraId, mCameraInfo);
			setDefaultParameters();
		} catch (Exception e) {
			mCamera = null;
			Log.i(TAG, "openCamera fail msg=" + e.getMessage());
			return false;
		}
		return true;
	}

	public void releaseCamera() {
		if (mCamera != null) {
			mCamera.setPreviewCallback(null);
			mCamera.stopPreview();
			mCamera.release();
			mCamera = null;
		}
	}

	public void startPreview(SurfaceTexture surfaceTexture, PreviewCallback previewcallback) {
		try {
			mCamera.setPreviewTexture(surfaceTexture);
			if (previewcallback != null) {
				mCamera.setPreviewCallback(previewcallback);
			}
			mCamera.startPreview();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public void startPreview(SurfaceHolder surfaceHolder, PreviewCallback previewcallback) {
		try {
			mCamera.setPreviewDisplay(surfaceHolder);
			if (previewcallback != null) {
				mCamera.setPreviewCallback(previewcallback);
			}
			mCamera.startPreview();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public void stopPreview() {
		if (mCamera != null)
			mCamera.stopPreview();
	}

	public void addCallbackBuffer(byte[] previewBuffer) {
        mCamera.addCallbackBuffer(previewBuffer);
    }



	/*
	 * preview size
	 */
	public void setPreviewSize(int width, int height) {
		if (mCamera == null) {
            return;
        }

		Parameters parameters = mCamera.getParameters();
		parameters.setPreviewSize(width, height);
        parameters.setPreviewFormat(ImageFormat.NV21);
		mCamera.setParameters(parameters);
	}

	private Point getSuitablePreviewSize() {
		Point defaultsize = new Point(1920, 1080);
		if (mCamera != null) {
			List<Size> sizes = mCamera.getParameters().getSupportedPreviewSizes();
			for (Size s : sizes) {
				if ((s.width == defaultsize.x) && (s.height == defaultsize.y)) {
					return defaultsize;
				}
			}
			return new Point(640, 480);
		}
		return null;
	}

	public ArrayList<String> getSupportedPreviewSize(String[] previewSizes) {
		ArrayList<String> result = new ArrayList<String>();
		if (mCamera != null) {
			List<Size> sizes = mCamera.getParameters().getSupportedPreviewSizes();
			for (String candidate : previewSizes) {
				int index = candidate.indexOf('x');
				if (index == -1) continue;
				int width = Integer.parseInt(candidate.substring(0, index));
				int height = Integer.parseInt(candidate.substring(index + 1));
				for (Size s : sizes) {
					if ((s.width == width) && (s.height == height)) {
						result.add(candidate);
					}
				}
			}
		}
		return result;
	}

    private Point getSuitablePictureSize() {
		Point defaultsize = new Point(4608, 3456);
		//	Point defaultsize = new Point(3264, 2448);
		if (mCamera != null) {
			Point maxSize = new Point(0, 0);
			List<Size> sizes = mCamera.getParameters().getSupportedPictureSizes();
			for (Size s : sizes) {
				if ((s.width == defaultsize.x) && (s.height == defaultsize.y)) {
					return defaultsize;
				}
				if (maxSize.x < s.width) {
					maxSize.x = s.width;
					maxSize.y = s.height;
				}
			}
			return maxSize;
		}
		return null;
	}


    /*
     * miscellaneous
     */
	private void setDefaultParameters() {
		Parameters parameters = mCamera.getParameters();

        if (parameters.getSupportedFocusModes().contains(Parameters.FOCUS_MODE_CONTINUOUS_VIDEO)) {
                parameters.setFocusMode(Parameters.FOCUS_MODE_CONTINUOUS_VIDEO);
        }
		List<String> flashModes = parameters.getSupportedFlashModes();
		if (flashModes != null && flashModes.contains(Parameters.FLASH_MODE_OFF)) {
			parameters.setFlashMode(Parameters.FLASH_MODE_OFF);
		}

		parameters.setPreviewSize(640, 480);
		Point pictureSize = getSuitablePictureSize();
		parameters.setPictureSize(pictureSize.x, pictureSize.y);

		mCamera.setParameters(parameters);
	}

	public void setCameraFocus(Parameters parameters) {
	    try {
            if (parameters.getSupportedFocusModes().contains(Parameters.FOCUS_MODE_AUTO)) {
                parameters.setFocusMode(Parameters.FOCUS_MODE_AUTO);

                mCamera.autoFocus((success, camera) -> {
                    if (success) {
                        Parameters param = camera.getParameters();

                        float[] focusDist = new float[3];
                        param.getFocusDistances(focusDist);
                        float focalLength = param.getFocalLength();

                        Log.e(TAG, String.format("*** focal length: %f, focusDist: %s ***",
                            focalLength, Arrays.toString(focusDist)));
                    }
                });
                return;
            }

            if (parameters.getSupportedFocusModes().contains(Parameters.FOCUS_MODE_CONTINUOUS_VIDEO)) {
                parameters.setFocusMode(Parameters.FOCUS_MODE_CONTINUOUS_VIDEO);
                return;
            }

            Log.e(TAG, "parameters: " + parameters.flatten());
        }
        catch (Exception e) {
            Log.e(TAG, ExceptionUtils.getStackTrace(e));
        }
    }

	public void setRotation(int rotation) {
		if (mCamera == null) {
            return;
        }

        Camera.Parameters params = mCamera.getParameters();
        params.setRotation(rotation);
        mCamera.setParameters(params);
        mCamera.setDisplayOrientation(rotation);
	}

    public boolean isFlipHorizontal() {
		if (mCameraInfo == null) {
			return false;
		}

		return mCameraInfo.facing == CameraInfo.CAMERA_FACING_FRONT;
	}

	public boolean isFrontCamera() {
		return mCameraId == CameraInfo.CAMERA_FACING_FRONT;
	}

    public int getCameraId() {
        return mCameraId;
    }

	public Camera getCamera() {
		return mCamera;
	}

	public int getNumberOfCameras() {
		return Camera.getNumberOfCameras();
	}

	public float getAngleOfView() {
		return mCamera.getParameters().getVerticalViewAngle();
	}

	public Parameters getParameters() {
        return mCamera.getParameters();
    }

    public int getOrientation() {
        return mCameraInfo == null ? 0 : mCameraInfo.orientation;
    }

    public int getDisplayOrientation(int dir) {
		/**
		 * 请注意前置摄像头与后置摄像头旋转定义不同
		 * 请注意不同手机摄像头旋转定义不同
		 */
		int newdir = dir;
		if (isFrontCamera() && ((mCameraInfo.orientation == 270 && (dir & 1) == 1)
            || (mCameraInfo.orientation == 90 && (dir & 1) == 0))) {
			newdir = (dir ^ 2);
		}
		return newdir;
	}

    public boolean needMirror() {
	    return isFrontCamera();
	}

}