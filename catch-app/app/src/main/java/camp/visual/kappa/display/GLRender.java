package camp.visual.kappa.display;

import android.opengl.GLES11Ext;
import android.opengl.GLES20;
import android.opengl.Matrix;

import com.sensetime.stmobile.model.STPoint3f;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import camp.visual.kappa.glutils.GlUtil;
import camp.visual.kappa.glutils.OpenGLUtils;
import camp.visual.kappa.glutils.TextureRotationUtil;
import camp.visual.kappa.utils.LogUtils;


public class GLRender {

    private int first[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 };
    private int second[] = { 0, 0, 1, 2, 0, 4, 5, 6, 0, 8, 9, 10, 0, 12, 13, 14, 0, 16, 17, 18 };

    private final static String TAG = "GLRender";
    private static final String CAMERA_INPUT_VERTEX_SHADER = "" +
            "attribute vec4 position;\n" +
            "attribute vec4 inputTextureCoordinate;\n" +
            "\n" +
            "varying vec2 textureCoordinate;\n" +
            "\n" +
            "void main()\n" +
            "{\n" +
            "	textureCoordinate = inputTextureCoordinate.xy;\n" +
            "	gl_Position = position;\n" +
            "}";

    private static final String CAMERA_INPUT_FRAGMENT_SHADER_OES = "" +
            "#extension GL_OES_EGL_image_external : require\n" +
            "\n" +
            "precision mediump float;\n" +
            "varying vec2 textureCoordinate;\n" +
            "uniform samplerExternalOES inputImageTexture;\n" +
            "\n" +
            "void main()\n" +
            "{\n" +
            "	gl_FragColor = texture2D(inputImageTexture, textureCoordinate);\n" +
            "}";

    public static final String CAMERA_INPUT_FRAGMENT_SHADER = "" +
            "precision mediump float;\n" +
            "varying highp vec2 textureCoordinate;\n" +
            " \n" +
            "uniform sampler2D inputImageTexture;\n" +
            " \n" +
            "void main()\n" +
            "{\n" +
            "     gl_FragColor = texture2D(inputImageTexture, textureCoordinate);\n" +
            "}";

    public static final String DRAW_POINTS_VERTEX_SHADER = "" +
            "attribute vec4 aPosition;\n" +
            "void main() {\n" +
            "  gl_PointSize = 1.0;" +
            "  gl_Position = aPosition;\n" +
            "}";

    public static final String DRAW_POINTS_FRAGMENT_SHADER = "" +
            "precision mediump float;\n" +
            "uniform vec4 uColor;\n" +
            "void main() {\n" +
            "  gl_FragColor = uColor;\n" +
            "}";

    public static final String DRAW_SEGMENT_FRAGMENT_SHADER = "" +
            "precision mediump float;\n" +
            "varying mediump vec2 textureCoordinate;\n" +
            "uniform sampler2D maskTexture;\n" +
            "uniform vec2 modifier;\n" +
            "uniform sampler2D backgroundTexture;\n" +
            "uniform vec3 edgeColor;\n" +
            "void main()\n" +
            "{\n" +
            "float maskValue = texture2D(maskTexture, textureCoordinate).r;\n" +
            "maskValue = maskValue * modifier.x + modifier.y;\n" +
            "if(modifier.x < 0.0){\n" +
            "maskValue *= 0.6;\n" +
            "};\n" +
            "vec4 backgroundColor = texture2D(backgroundTexture, textureCoordinate);\n" +
            "gl_FragColor = vec4(mix(backgroundColor.rgb, edgeColor, maskValue), backgroundColor.a);\n" +
            "}";

    public static final String DRAW_3DHAND_VERTEX_SHADER = "" +
            "attribute vec3 position;\n" +
            "uniform mat4 model;\n" +
            "uniform mat4 view;\n" +
            "uniform mat4 projection;\n" +
            "void main()\n" +
            "{\n" +
            "	gl_Position = projection * view * model * vec4(position, 1.0);\n" +
            "   gl_PointSize = 10.0;\n" +
            "}";

    public static final String DRAW_3DHAND_FRAGMENT_SHADER = "" +
            "precision mediump float;\n" +
            "uniform vec4 color;\n" +
            "void main() {\n" +
            "  gl_FragColor = color;\n" +
            "}";

    //
    private final static String DRAW_POINTS_PROGRAM = "mPointProgram";
    private final static String DRAW_POINTS_COLOR = "uColor";
    private final static String DRAW_POINTS_POSITION = "aPosition";
    private int mDrawPointsProgram = 0;
    private int mColor = -1;
    private int mPosition = -1;
    private int[] mPointsFrameBuffers;

    //3DHand
    private final static String HAND_VERTEX_POSITION = "position";
    private final static String MODEL_MATRIX = "model";
    private final static String VIEW_MATRIX = "view";
    private final static String PROJECTION_MATRIX = "projection";
    private final static String COLOR = "color";
    private int mDraw3DHandProgram = 0;
    private int mHandVertexPosition = -1;
    private int mModelMatrix = -1;
    private int mViewMatrix = -1;
    private int mProjectionMatrix = -1;
    private int m3DHandColor = -1;
    private float[] modelMatrix = new float[16];
    private float[] viewMatrix = new float[16];
    private float[] projectionMatrix = new float[16];

    //segment
    private final static String TEXTURE_BACKGROUND = "backgroundTexture";
    private final static String TEXTURE_MASK = "maskTexture";
    private final static String TEXTURE_MODIFIER = "modifier";
    private final static String TEXTURE_EDGECOLOR = "edgeColor";
    private int[] mBackgroundFrameBuffers;
    private int[] mHairFrameBuffers;
    private int mDrawBackGroundProgram = 0;
    private int mAttribVertex = -1;
    private int mTexturePosition = -1;
    private int mBackground = -1;
    private int mMask = -1;
    private int mModifier = -1;
    private int mEdgeColor = -1;
    private int mFigureSegmentResultTexture = OpenGLUtils.NO_TEXTURE;
    private int mHairSegmentResultTexture = OpenGLUtils.NO_TEXTURE;
    private FloatBuffer mSegmentVertexBuffer;

    private final static String PROGRAM_ID = "program";
    private final static String POSITION_COORDINATE = "position";
    private final static String TEXTURE_UNIFORM = "inputImageTexture";
    private final static String TEXTURE_COORDINATE = "inputTextureCoordinate";
    private final FloatBuffer mGLCubeBuffer;
    private final FloatBuffer mGLTextureBuffer;
    private final FloatBuffer mGLSaveTextureBuffer;
    private FloatBuffer m3DHandBuffer;
    private FloatBuffer m3DHandLineBuffer;
    private FloatBuffer m3DLineBuffer;

    private FloatBuffer mTextureBuffer;
    private FloatBuffer mVertexBuffer;

    private boolean mIsInitialized;
    private ArrayList<HashMap<String, Integer>> mArrayPrograms = new ArrayList<HashMap<String, Integer>>(2) {
        {
            for (int i = 0; i < 2; ++i) {
                HashMap<String, Integer> hashMap = new HashMap<>();
                hashMap.put(PROGRAM_ID, 0);
                hashMap.put(POSITION_COORDINATE, -1);
                hashMap.put(TEXTURE_UNIFORM, -1);
                hashMap.put(TEXTURE_COORDINATE, -1);
                add(hashMap);
            }
        }
    };
    private int mViewPortWidth;
    private int mViewPortHeight;
    private int[] mFrameBuffers;
    private int[] mFrameBufferTextures;


    public GLRender() {
        mGLCubeBuffer = ByteBuffer.allocateDirect(TextureRotationUtil.CUBE.length * 4)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer();
        mGLCubeBuffer.put(TextureRotationUtil.CUBE).position(0);

        mGLTextureBuffer = ByteBuffer.allocateDirect(TextureRotationUtil.TEXTURE_NO_ROTATION.length * 4)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer();
        mGLTextureBuffer.put(TextureRotationUtil.TEXTURE_NO_ROTATION).position(0);

        mGLSaveTextureBuffer = ByteBuffer.allocateDirect(TextureRotationUtil.TEXTURE_NO_ROTATION.length * 4)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer();
        mGLSaveTextureBuffer.put(TextureRotationUtil.getRotation(0, false, true)).position(0);

        mSegmentVertexBuffer = ByteBuffer.allocateDirect(TextureRotationUtil.CUBE.length * 4)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer();
        mSegmentVertexBuffer.put(TextureRotationUtil.cubeFlipVertical()).position(0);
    }

    public void init(int width, int height) {
        if (mViewPortWidth == width && mViewPortHeight == height) {
            return ;
        }
        initProgram(CAMERA_INPUT_FRAGMENT_SHADER_OES, mArrayPrograms.get(0));
        initProgram(CAMERA_INPUT_FRAGMENT_SHADER, mArrayPrograms.get(1));
        mViewPortWidth = width;
        mViewPortHeight = height;
        initFrameBuffers(width, height);
        mIsInitialized = true;
    }

    private void initProgram(String fragment, HashMap<String, Integer> programInfo) {
        int proID = programInfo.get(PROGRAM_ID);
        if (proID == 0) {
            proID = OpenGLUtils.loadProgram(CAMERA_INPUT_VERTEX_SHADER, fragment);
            programInfo.put(PROGRAM_ID, proID);
            programInfo.put(POSITION_COORDINATE, GLES20.glGetAttribLocation(proID, POSITION_COORDINATE));
            programInfo.put(TEXTURE_UNIFORM, GLES20.glGetUniformLocation(proID, TEXTURE_UNIFORM));
            programInfo.put(TEXTURE_COORDINATE, GLES20.glGetAttribLocation(proID, TEXTURE_COORDINATE));
        }
    }

    public void initDrawPoints() {
        mDrawPointsProgram = OpenGLUtils.loadProgram(DRAW_POINTS_VERTEX_SHADER, DRAW_POINTS_FRAGMENT_SHADER);
        mColor = GLES20.glGetAttribLocation(mDrawPointsProgram, DRAW_POINTS_POSITION);
        mPosition = GLES20.glGetUniformLocation(mDrawPointsProgram, DRAW_POINTS_COLOR);

        if (mPointsFrameBuffers == null) {
            mPointsFrameBuffers = new int[1];

            GLES20.glGenFramebuffers(1, mPointsFrameBuffers, 0);
        }
    }

    public void initDraw3DHand() {
        mDraw3DHandProgram = OpenGLUtils.loadProgram(DRAW_3DHAND_VERTEX_SHADER, DRAW_3DHAND_FRAGMENT_SHADER);
        mHandVertexPosition = GLES20.glGetAttribLocation(mDraw3DHandProgram, HAND_VERTEX_POSITION);
        mModelMatrix = GLES20.glGetUniformLocation(mDraw3DHandProgram, MODEL_MATRIX);
        mViewMatrix = GLES20.glGetUniformLocation(mDraw3DHandProgram, VIEW_MATRIX);
        mProjectionMatrix = GLES20.glGetUniformLocation(mDraw3DHandProgram, PROJECTION_MATRIX);
        m3DHandColor = GLES20.glGetUniformLocation(mDraw3DHandProgram, COLOR);
    }

    public void initDrawBackGround() {
        mDrawBackGroundProgram = OpenGLUtils.loadProgram(CAMERA_INPUT_VERTEX_SHADER, DRAW_SEGMENT_FRAGMENT_SHADER);
        mAttribVertex = GLES20.glGetAttribLocation(mDrawBackGroundProgram, POSITION_COORDINATE);
        mTexturePosition = GLES20.glGetAttribLocation(mDrawBackGroundProgram, TEXTURE_COORDINATE);
        mBackground = GLES20.glGetUniformLocation(mDrawBackGroundProgram, TEXTURE_BACKGROUND);
        mMask = GLES20.glGetUniformLocation(mDrawBackGroundProgram, TEXTURE_MASK);
        mModifier = GLES20.glGetUniformLocation(mDrawBackGroundProgram, TEXTURE_MODIFIER);
        mEdgeColor = GLES20.glGetUniformLocation(mDrawBackGroundProgram, TEXTURE_EDGECOLOR);

        if (mBackgroundFrameBuffers == null) {
            mBackgroundFrameBuffers = new int[1];

            GLES20.glGenFramebuffers(1, mBackgroundFrameBuffers, 0);
        }

        if (mHairFrameBuffers == null) {
            mHairFrameBuffers = new int[1];

            GLES20.glGenFramebuffers(1, mHairFrameBuffers, 0);
        }
    }

    public void adjustTextureBuffer(int orientation, boolean flipVertical) {
        float[] textureCords = TextureRotationUtil.getRotation(orientation, true, flipVertical);
        LogUtils.d(TAG, "==========rotation: " + orientation + " flipVertical: " + flipVertical
                + " texturePos: " + Arrays.toString(textureCords));
        if (mTextureBuffer == null) {
            mTextureBuffer = ByteBuffer.allocateDirect(textureCords.length * 4)
                    .order(ByteOrder.nativeOrder())
                    .asFloatBuffer();
        }
        mTextureBuffer.clear();
        mTextureBuffer.put(textureCords).position(0);
    }

    /**
     * 用来计算贴纸渲染的纹理最终需要的顶点坐标
     */
    public void calculateVertexBuffer(int displayW, int displayH, int imageW, int imageH) {
        int outputHeight = displayH;
        int outputWidth = displayW;

        float ratio1 = (float) outputWidth / imageW;
        float ratio2 = (float) outputHeight / imageH;
        float ratioMin = Math.min(ratio1, ratio2);
        int imageWidthNew = Math.round(imageW * ratioMin);
        int imageHeightNew = Math.round(imageH * ratioMin);

        float ratioWidth = imageWidthNew / (float) outputWidth;
        float ratioHeight = imageHeightNew / (float) outputHeight;

        float[] cube = new float[]{
                TextureRotationUtil.CUBE[0] / ratioHeight, TextureRotationUtil.CUBE[1] / ratioWidth,
                TextureRotationUtil.CUBE[2] / ratioHeight, TextureRotationUtil.CUBE[3] / ratioWidth,
                TextureRotationUtil.CUBE[4] / ratioHeight, TextureRotationUtil.CUBE[5] / ratioWidth,
                TextureRotationUtil.CUBE[6] / ratioHeight, TextureRotationUtil.CUBE[7] / ratioWidth,
        };

        if (mVertexBuffer == null) {
            mVertexBuffer = ByteBuffer.allocateDirect(cube.length * 4)
                    .order(ByteOrder.nativeOrder())
                    .asFloatBuffer();
        }
        mVertexBuffer.clear();
        mVertexBuffer.put(cube).position(0);
    }

    /**
     * 此函数有三个功能
     * 1. 将OES的纹理转换为标准的GL_TEXTURE_2D格式
     * 2. 将纹理宽高对换，即将wxh的纹理转换为了hxw的纹理，并且如果是前置摄像头，则需要有水平的翻转
     * 3. 读取上面两个步骤后纹理的内容到cpu内存，存储为RGBA格式的buffer
     * @param textureId 输入的OES的纹理id
     * @param buffer 输出的RGBA的buffer
     * @return 转换后的GL_TEXTURE_2D的纹理id
     */
    public int preProcess(int textureId, ByteBuffer buffer) {
        if (mFrameBuffers == null
                || !mIsInitialized)
            return -2;

        GLES20.glUseProgram(mArrayPrograms.get(0).get(PROGRAM_ID));
        GlUtil.checkGlError("glUseProgram");

        mGLCubeBuffer.position(0);
        int glAttribPosition = mArrayPrograms.get(0).get(POSITION_COORDINATE);
        GLES20.glVertexAttribPointer(glAttribPosition, 2, GLES20.GL_FLOAT, false, 0, mGLCubeBuffer);
        GLES20.glEnableVertexAttribArray(glAttribPosition);

        mTextureBuffer.position(0);
        int glAttribTextureCoordinate = mArrayPrograms.get(0).get(TEXTURE_COORDINATE);
        GLES20.glVertexAttribPointer(glAttribTextureCoordinate, 2, GLES20.GL_FLOAT, false, 0, mTextureBuffer);
        GLES20.glEnableVertexAttribArray(glAttribTextureCoordinate);

        if (textureId != -1) {
            GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
            GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textureId);
            GLES20.glUniform1i(mArrayPrograms.get(0).get(TEXTURE_UNIFORM), 0);
        }
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, mFrameBuffers[0]);
        GlUtil.checkGlError("glBindFramebuffer");
        GLES20.glViewport(0, 0, mViewPortWidth, mViewPortHeight);

        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);

        if (buffer != null) {
            GLES20.glReadPixels(0, 0, mViewPortWidth, mViewPortHeight, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, buffer);
        }

        GLES20.glDisableVertexAttribArray(glAttribPosition);
        GLES20.glDisableVertexAttribArray(glAttribTextureCoordinate);
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, 0);

        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);
        GLES20.glUseProgram(0);

        return mFrameBufferTextures[0];
    }

    public void destroyFrameBuffers() {
        if (mFrameBufferTextures != null) {
            GLES20.glDeleteTextures(2, mFrameBufferTextures, 0);
            mFrameBufferTextures = null;
        }
        if (mFrameBuffers != null) {
            GLES20.glDeleteFramebuffers(2, mFrameBuffers, 0);
            mFrameBuffers = null;
        }

        if (mPointsFrameBuffers != null) {
            GLES20.glDeleteFramebuffers(1, mPointsFrameBuffers, 0);
            mPointsFrameBuffers = null;
        }

        if (mBackgroundFrameBuffers != null) {
            GLES20.glDeleteFramebuffers(1, mBackgroundFrameBuffers, 0);
            mBackgroundFrameBuffers = null;
        }

        if (mHairFrameBuffers != null) {
            GLES20.glDeleteFramebuffers(1, mHairFrameBuffers, 0);
            mHairFrameBuffers = null;
        }
    }

    public void onDrawPoints(int textureId, float[] points) {

        if (mDrawPointsProgram == 0) {
            initDrawPoints();
        }

        GLES20.glUseProgram(mDrawPointsProgram);
        GLES20.glUniform4f(mColor, 0.0f, 0.0f, 0.0f, 0.0f);

        FloatBuffer buff = null;

        buff = ByteBuffer.allocateDirect(points.length * 4)
                .order(ByteOrder.nativeOrder()).asFloatBuffer();

        buff.clear();
        buff.put(points).position(0);

        GLES20.glVertexAttribPointer(mPosition, 2, GLES20.GL_FLOAT, false, 0, buff);
        GLES20.glEnableVertexAttribArray(mPosition);

        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, mPointsFrameBuffers[0]);
        GLES20.glFramebufferTexture2D(GLES20.GL_FRAMEBUFFER, GLES20.GL_COLOR_ATTACHMENT0,
                GLES20.GL_TEXTURE_2D, textureId, 0);

        GlUtil.checkGlError("glBindFramebuffer");
        GLES20.glViewport(0, 0, mViewPortWidth, mViewPortHeight);

        GLES20.glDrawArrays(GLES20.GL_POINTS, 0, points.length/2);

        GLES20.glDisableVertexAttribArray(mPosition);

    }

    public static int genFrameBufferTextureID(int width, int height){
        int[] texture = new int[1];

        GLES20.glGenTextures(1, texture, 0);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, texture[0]);

        GLES20.glTexParameterf(GLES20.GL_TEXTURE_2D,
                GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameterf(GLES20.GL_TEXTURE_2D,
                GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameterf(GLES20.GL_TEXTURE_2D,
                GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameterf(GLES20.GL_TEXTURE_2D,
                GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES20.GL_RGBA, width, height, 0,
                GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, null);
        return texture[0];
    }

    public void initFigureSegmentResultTexture(){
        mFigureSegmentResultTexture = genFrameBufferTextureID(mViewPortWidth, mViewPortHeight);
    }

    public void initHairSegmentResultTexture(){
        mHairSegmentResultTexture = genFrameBufferTextureID(mViewPortWidth, mViewPortHeight);
    }

    public int onDrawSegmentImageToTexture(final int inputTexture, int maskTexture, int width, int height, ByteBuffer data, boolean isHairSegment) {

        if (!mIsInitialized) {
            return OpenGLUtils.NOT_INIT;
        }

        if (mDrawBackGroundProgram == 0) {
            initDrawBackGround();
        }

        if(mFigureSegmentResultTexture == OpenGLUtils.NO_TEXTURE && !isHairSegment){
            initFigureSegmentResultTexture();
        }

        if(mHairSegmentResultTexture == OpenGLUtils.NO_TEXTURE && isHairSegment){
            initHairSegmentResultTexture();
        }

        GLES20.glUseProgram(mDrawBackGroundProgram);
        GlUtil.checkGlError("glUseProgram");
        GLES20.glPixelStorei(GLES20.GL_UNPACK_ALIGNMENT, 1);

        GLES20.glActiveTexture(GLES20.GL_TEXTURE1);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, maskTexture);
        GLES20.glTexSubImage2D(GLES20.GL_TEXTURE_2D, 0, 0, 0, width, height, GLES20.GL_LUMINANCE, GLES20.GL_UNSIGNED_BYTE, data);

        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, inputTexture);

        GLES20.glUniform1i(mBackground, 0);
        GLES20.glUniform1i(mMask, 1);

        float edgeColor[] = {0.0f, 0.0f, 0.0f};
        float modifier[] = {1.0f, 0.0f};

        if (isHairSegment) {
            edgeColor[0] = 0.235f;
            edgeColor[1] = 0.145f;
            edgeColor[2] = 0.059f;

            modifier[0] = -1.0f;
            modifier[1] = 1.0f;
        }

        FloatBuffer edgeColorBuff = null;
        edgeColorBuff = ByteBuffer.allocateDirect(edgeColor.length * 4)
                .order(ByteOrder.nativeOrder()).asFloatBuffer();

        edgeColorBuff.clear();
        edgeColorBuff.put(edgeColor).position(0);
        GLES20.glUniform3fv(mEdgeColor, 1, edgeColorBuff);

        FloatBuffer modifierBuff = null;
        modifierBuff = ByteBuffer.allocateDirect(modifier.length * 4)
                .order(ByteOrder.nativeOrder()).asFloatBuffer();

        modifierBuff.clear();
        modifierBuff.put(modifier).position(0);
        GLES20.glUniform2fv(mModifier, 1, modifierBuff);

        GLES20.glActiveTexture(GLES20.GL_TEXTURE2);

        if(isHairSegment){
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, mHairSegmentResultTexture);
        }else {
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, mFigureSegmentResultTexture);
        }


        GlUtil.checkGlError("glBindFramebuffer");

        if(isHairSegment){
            GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, mHairFrameBuffers[0]);
            GLES20.glFramebufferTexture2D(GLES20.GL_FRAMEBUFFER, GLES20.GL_COLOR_ATTACHMENT0, GLES20.GL_TEXTURE_2D, mHairSegmentResultTexture, 0);
        }else {
            GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, mBackgroundFrameBuffers[0]);
            GLES20.glFramebufferTexture2D(GLES20.GL_FRAMEBUFFER, GLES20.GL_COLOR_ATTACHMENT0, GLES20.GL_TEXTURE_2D, mFigureSegmentResultTexture, 0);
        }

        mSegmentVertexBuffer.position(0);
        GLES20.glVertexAttribPointer(mAttribVertex, 2, GLES20.GL_FLOAT, false, 0, mSegmentVertexBuffer);
        GLES20.glEnableVertexAttribArray(mAttribVertex);

        mGLTextureBuffer.position(0);
        GLES20.glVertexAttribPointer(mTexturePosition, 2, GLES20.GL_FLOAT, false, 0, mGLTextureBuffer);
        GLES20.glEnableVertexAttribArray(mTexturePosition);
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);

        GLES20.glDisableVertexAttribArray(mAttribVertex);
        GLES20.glDisableVertexAttribArray(mTexturePosition);
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, 0);

        if(isHairSegment){
            return mHairSegmentResultTexture;
        }else {
            return mFigureSegmentResultTexture;
        }
    }

    public int onDrawFrame(final int textureId) {

        if (!mIsInitialized) {
            return OpenGLUtils.NOT_INIT;
        }

        GLES20.glUseProgram(mArrayPrograms.get(1).get(PROGRAM_ID));

        mVertexBuffer.position(0);
        int glAttribPosition = mArrayPrograms.get(1).get(POSITION_COORDINATE);
        GLES20.glVertexAttribPointer(glAttribPosition, 2, GLES20.GL_FLOAT, false, 0, mVertexBuffer);
        GLES20.glEnableVertexAttribArray(glAttribPosition);

        mGLTextureBuffer.position(0);
        int glAttribTextureCoordinate = mArrayPrograms.get(1).get(TEXTURE_COORDINATE);
        GLES20.glVertexAttribPointer(glAttribTextureCoordinate, 2, GLES20.GL_FLOAT, false, 0,
                mGLTextureBuffer);
        GLES20.glEnableVertexAttribArray(glAttribTextureCoordinate);

        if (textureId != OpenGLUtils.NO_TEXTURE) {
            GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, textureId);
            GLES20.glUniform1i(mArrayPrograms.get(1).get(TEXTURE_UNIFORM), 0);
        }

        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);
        GLES20.glDisableVertexAttribArray(glAttribPosition);
        GLES20.glDisableVertexAttribArray(glAttribTextureCoordinate);
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, 0);
        return OpenGLUtils.ON_DRAWN;
    }

    public int saveTextureToFrameBuffer(int textureOutId, ByteBuffer buffer) {
        if(mFrameBuffers == null) {
            return OpenGLUtils.NO_TEXTURE;
        }
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, mFrameBuffers[1]);
        GLES20.glViewport(0, 0, mViewPortWidth, mViewPortHeight);

        GLES20.glUseProgram(mArrayPrograms.get(1).get(PROGRAM_ID));

        if(!mIsInitialized) {
            return OpenGLUtils.NOT_INIT;
        }

        mGLCubeBuffer.position(0);
        int glAttribPosition = mArrayPrograms.get(1).get(POSITION_COORDINATE);
        GLES20.glVertexAttribPointer(glAttribPosition, 2, GLES20.GL_FLOAT, false, 0, mGLCubeBuffer);
        GLES20.glEnableVertexAttribArray(glAttribPosition);

        mGLSaveTextureBuffer.position(0);
        int glAttribTextureCoordinate = mArrayPrograms.get(1).get(TEXTURE_COORDINATE);
        GLES20.glVertexAttribPointer(glAttribTextureCoordinate, 2, GLES20.GL_FLOAT, false, 0, mGLSaveTextureBuffer);
        GLES20.glEnableVertexAttribArray(glAttribTextureCoordinate);

        if(textureOutId != OpenGLUtils.NO_TEXTURE) {
            GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, textureOutId);
            GLES20.glUniform1i(mArrayPrograms.get(1).get(TEXTURE_UNIFORM), 0);
        }

        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);

        if(buffer != null) {
            GLES20.glReadPixels(0, 0, mViewPortWidth, mViewPortHeight, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, buffer);
        }

        GLES20.glDisableVertexAttribArray(glAttribPosition);
        GLES20.glDisableVertexAttribArray(glAttribTextureCoordinate);

        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, 0);
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);

        return mFrameBufferTextures[1];
    }

    private void initFrameBuffers(int width, int height) {
        destroyFrameBuffers();

        if (mFrameBuffers == null) {
            mFrameBuffers = new int[2];
            mFrameBufferTextures = new int[2];

            GLES20.glGenFramebuffers(2, mFrameBuffers, 0);
            GLES20.glGenTextures(2, mFrameBufferTextures, 0);

            bindFrameBuffer(mFrameBufferTextures[0], mFrameBuffers[0], width, height);
            bindFrameBuffer(mFrameBufferTextures[1], mFrameBuffers[1], width, height);
        }
    }

    private void bindFrameBuffer(int textureId, int frameBuffer, int width, int height) {
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, textureId);
        GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES20.GL_RGBA, width, height, 0,
                GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, null);
        GLES20.glTexParameterf(GLES20.GL_TEXTURE_2D,
                GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameterf(GLES20.GL_TEXTURE_2D,
                GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameterf(GLES20.GL_TEXTURE_2D,
                GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameterf(GLES20.GL_TEXTURE_2D,
                GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);

        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, frameBuffer);
        GLES20.glFramebufferTexture2D(GLES20.GL_FRAMEBUFFER, GLES20.GL_COLOR_ATTACHMENT0,
                GLES20.GL_TEXTURE_2D,textureId, 0);

        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, 0);
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);
    }

    public final void destroy() {
        mIsInitialized = false;
        destroyFrameBuffers();
        GLES20.glDeleteProgram(mArrayPrograms.get(0).get(PROGRAM_ID));
        GLES20.glDeleteProgram(mArrayPrograms.get(1).get(PROGRAM_ID));
    }

    public void deleteSegmentResultTexture(){
        if (mFigureSegmentResultTexture != OpenGLUtils.NO_TEXTURE) {
            GLES20.glDeleteTextures(1, new int[]{mFigureSegmentResultTexture}, 0);
        }
        mFigureSegmentResultTexture = OpenGLUtils.NO_TEXTURE;

        if (mHairSegmentResultTexture != OpenGLUtils.NO_TEXTURE) {
            GLES20.glDeleteTextures(1, new int[]{mHairSegmentResultTexture}, 0);
        }
        mHairSegmentResultTexture = OpenGLUtils.NO_TEXTURE;
    }

    public void destroySegmentFrameBuffer(){
        if (mBackgroundFrameBuffers != null) {
            GLES20.glDeleteFramebuffers(1, mBackgroundFrameBuffers, 0);
            mBackgroundFrameBuffers = null;
        }

        if (mHairFrameBuffers != null) {
            GLES20.glDeleteFramebuffers(1, mHairFrameBuffers, 0);
            mHairFrameBuffers = null;
        }
    }

    public void onDrawGrid(int imageWidth, int imageHeight) {
        if (mDraw3DHandProgram == 0) {
            initDraw3DHand();
        }
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, mFrameBuffers[0]);
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT | GLES20.GL_DEPTH_BUFFER_BIT);

//        if(m3DLineBuffer == null){
//            m3DLineBuffer = ByteBuffer.allocateDirect(3 * 2 * 4)
//                    .order(ByteOrder.nativeOrder())
//                    .asFloatBuffer();
//        }
//
//        GLES20.glUseProgram(mDraw3DHandProgram);
//
//        Matrix.setIdentityM(modelMatrix,0);
//        GLES20.glUniformMatrix4fv(mModelMatrix, 1, false, modelMatrix,0);
//
//        Matrix.setLookAtM(viewMatrix,0, 0f,0f,2f,0f,0f,-1f,0f,1f,0f);
//        GLES20.glUniformMatrix4fv(mViewMatrix, 1, false, viewMatrix,0);
//
//        Matrix.perspectiveM(projectionMatrix, 0,45.0f, (float)imageWidth/(float) imageHeight,0.1f,100.0f);
//        GLES20.glUniformMatrix4fv(mProjectionMatrix, 1, false, projectionMatrix,0);
//
//        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, mFrameBuffers[0]);
//        GLES20.glUniform4f(m3DHandColor, 1.0f, 0.0f, 0.0f, 1.0f);
//        GLES20.glEnable(GLES20.GL_DEPTH_TEST);
//        GLES20.glViewport(0, 0, mViewPortWidth, mViewPortHeight);
//
//        drawSpaceLine();
//
//        GLES20.glDisable(GLES20.GL_DEPTH_TEST);
//        GLES20.glDisableVertexAttribArray(mHandVertexPosition);
//        GLES20.glUseProgram(0);
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);

    }

    private void drawSpaceLine(){
        GLES20.glUniform4f(m3DHandColor, 0.0f, 1.0f, 0.0f, 1.0f);
        for(float i = 1; i <= 10; i++){
            m3DLineBuffer.put(-1.0f);
            m3DLineBuffer.put(i * 0.2f - 1f);
            m3DLineBuffer.put(-50.0f);
            m3DLineBuffer.put(1.0f);
            m3DLineBuffer.put(i * 0.2f - 1f);
            m3DLineBuffer.put(-50.0f);
            m3DLineBuffer.position(0);
            GLES20.glVertexAttribPointer(mHandVertexPosition, 3, GLES20.GL_FLOAT, false, 0, m3DLineBuffer);
            GLES20.glEnableVertexAttribArray(mHandVertexPosition);
            GLES20.glDrawArrays(GLES20.GL_LINES, 0, 2);
        }
//        for(float i = 1; i <= 10; i++){
//            m3DLineBuffer.put(-1.0f);
//            m3DLineBuffer.put(i * 0.2f - 1f);
//            m3DLineBuffer.put(-1.0f);
//            m3DLineBuffer.put(1.0f);
//            m3DLineBuffer.put(i * 0.2f - 1f);
//            m3DLineBuffer.put(-1.0f);
//            m3DLineBuffer.position(0);
//            GLES20.glVertexAttribPointer(mHandVertexPosition, 3, GLES20.GL_FLOAT, false, 0, m3DLineBuffer);
//            GLES20.glEnableVertexAttribArray(mHandVertexPosition);
//            GLES20.glDrawArrays(GLES20.GL_LINES, 0, 2);
//        }
    }

    public int onDraw3DHand(int imageWidth, int imageHeight, STPoint3f[] extra3DKeyPoints){
        if (mDraw3DHandProgram == 0) {
            initDraw3DHand();
        }
        init3DHandBuffer(extra3DKeyPoints);

        GLES20.glUseProgram(mDraw3DHandProgram);

        m3DHandBuffer.position(0);
        GLES20.glVertexAttribPointer(mHandVertexPosition, 3, GLES20.GL_FLOAT, false, 0, m3DHandBuffer);
        GLES20.glEnableVertexAttribArray(mHandVertexPosition);

        Matrix.setIdentityM(modelMatrix,0);
//        Matrix.translateM(modelMatrix,0,0f,0f,0f);
        GLES20.glUniformMatrix4fv(mModelMatrix, 1, false, modelMatrix,0);

        Matrix.setLookAtM(viewMatrix,0, 0f,0f,0.3f,0f,0f,-1f,0f,1f,0f);
        GLES20.glUniformMatrix4fv(mViewMatrix, 1, false, viewMatrix,0);

        Matrix.perspectiveM(projectionMatrix, 0,45.0f, (float)imageWidth/(float) imageHeight,0.1f,100.0f);
        GLES20.glUniformMatrix4fv(mProjectionMatrix, 1, false, projectionMatrix,0);

        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, mFrameBuffers[0]);
        GLES20.glUniform4f(m3DHandColor, 1.0f, 0.0f, 0.0f, 1.0f);
        GLES20.glEnable(GLES20.GL_DEPTH_TEST);
        GLES20.glViewport(0, 0, mViewPortWidth, mViewPortHeight);
        GLES20.glDrawArrays(GLES20.GL_POINTS, 0, extra3DKeyPoints.length);

        GLES20.glUniform4f(m3DHandColor, 0.0f, 1.0f, 0.0f, 1.0f);
        for(int i = 0; i < 20; i++){
            m3DHandLineBuffer.clear();
            m3DHandLineBuffer.put(extra3DKeyPoints[first[i]].getX());
            m3DHandLineBuffer.put(extra3DKeyPoints[first[i]].getY());
            m3DHandLineBuffer.put(-extra3DKeyPoints[first[i]].getZ());
            m3DHandLineBuffer.put(extra3DKeyPoints[second[i]].getX());
            m3DHandLineBuffer.put(extra3DKeyPoints[second[i]].getY());
            m3DHandLineBuffer.put(-extra3DKeyPoints[second[i]].getZ());
            m3DHandLineBuffer.position(0);
            GLES20.glVertexAttribPointer(mHandVertexPosition, 3, GLES20.GL_FLOAT, false, 0, m3DHandLineBuffer);
            GLES20.glEnableVertexAttribArray(mHandVertexPosition);
            GLES20.glDrawArrays(GLES20.GL_LINES, 0, 2);
        }
        GLES20.glDisable(GLES20.GL_DEPTH_TEST);
        GLES20.glDisableVertexAttribArray(mHandVertexPosition);
        GLES20.glUseProgram(0);
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);
        return mFrameBufferTextures[0];
    }

    private void init3DHandBuffer(STPoint3f[] extra3DKeyPoints){
        if(m3DHandBuffer == null){
            m3DHandBuffer = ByteBuffer.allocateDirect(3 * extra3DKeyPoints.length * 4)
                    .order(ByteOrder.nativeOrder())
                    .asFloatBuffer();
        }
        float[] tmp = new float[extra3DKeyPoints.length * 3];
        for(int j = 0; j < extra3DKeyPoints.length; j++){
            tmp[j * 3] = extra3DKeyPoints[j].getX();
            tmp[j * 3 + 1] = extra3DKeyPoints[j].getY();
            tmp[j * 3 + 2] = -extra3DKeyPoints[j].getZ();
        }
        m3DHandBuffer.put(tmp).position(0);

        if(m3DHandLineBuffer == null){
            m3DHandLineBuffer = ByteBuffer.allocateDirect(3 * 2 * 4)
                    .order(ByteOrder.nativeOrder())
                    .asFloatBuffer();
        }
    }

}
