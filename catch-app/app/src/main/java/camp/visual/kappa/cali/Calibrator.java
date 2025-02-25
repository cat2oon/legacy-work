package camp.visual.kappa.cali;

import android.util.Log;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.analysis.MultivariateFunction;
import org.apache.commons.math3.analysis.MultivariateVectorFunction;
import org.apache.commons.math3.fitting.leastsquares.LeastSquaresBuilder;
import org.apache.commons.math3.fitting.leastsquares.LeastSquaresOptimizer;
import org.apache.commons.math3.fitting.leastsquares.LeastSquaresProblem;
import org.apache.commons.math3.fitting.leastsquares.LevenbergMarquardtOptimizer;
import org.apache.commons.math3.fitting.leastsquares.MultivariateJacobianFunction;
import org.apache.commons.math3.fitting.leastsquares.ParameterValidator;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.optim.BaseOptimizer;
import org.apache.commons.math3.optim.InitialGuess;
import org.apache.commons.math3.optim.MaxEval;
import org.apache.commons.math3.optim.PointValuePair;
import org.apache.commons.math3.optim.SimpleBounds;
import org.apache.commons.math3.optim.nonlinear.scalar.GoalType;
import org.apache.commons.math3.optim.nonlinear.scalar.LeastSquaresConverter;
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunction;
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer;
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv.PowellOptimizer;
import org.apache.commons.math3.util.Pair;
import org.eclipse.collections.impl.list.mutable.FastList;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Point3;

import java.util.Arrays;

import camp.visual.kappa.al.Face;
import camp.visual.kappa.ds.face.ZTPerspectiveMapper;
import camp.visual.kappa.zt.ZTDetectionParser;

import static camp.visual.ac.Maths.square;
import static camp.visual.ac.Maths.toDeg;
import static camp.visual.ac.Maths.toRad;
import static camp.visual.ac.Maths.toStringColVec;
import static camp.visual.ac.Maths.toStringMat3x3;
import static camp.visual.ac.cv2.CvHelper.toExtrinsicMat;
import static camp.visual.kappa.al.Face.computeEyeCenterOfRotationByCaruncle;
import static camp.visual.kappa.al.Face.computeFacePosition;
import static camp.visual.kappa.optimize.GazeOptimizer.makeAnchorCorrection;
import static camp.visual.kappa.optimize.distance.distanceL2;


public class Calibrator {
    private String TAG = "Calibrator";

    private Mat mRotVec = new Mat();
    private Mat mTransVec = new Mat();
    private FastList<Snapshot> mSnapshots;
    private ZTPerspectiveMapper mMapper;
    private ZTDetectionParser mDetectionParser;

    // Note: Powell Optimizer Bounds 허용 안함.
    private BaseOptimizer mDSMOptimizer;                // Direct Search Method
    private LeastSquaresOptimizer mOptimizer;           // Derivative, Jacobian
    private OptimalParamEstimator mEstimator;           // Param Learner

    // Domain Bounds
    private double mDxBound = 10.0;             // mm (좌/우)
    private double mDyBound = 10.0;             // mm (상/하)
    private double mDzBound = 20.0;             // mm (전/후) (R로 인해 부호 반대)
    private double mKappaBetaBound = 9;         // 통계 수집하여 정할 것
    private double mKappaAlphaBound = 9;        // 통계 수집하여 정할 것

    // Domain Initial
    private double mInitDx =  -12.0;
    private double mInitDy =   5.0;
    private double mInitDz =  -11.0;
    private double mInitKappaBeta = 3.0;
    private double mInitKappaAlpha = 3.0;

    // Optimization Process Holder
    private boolean mUsingDz = true;
    private boolean mUsingKappa = true;



    public Calibrator(ZTDetectionParser parser) {
        mDetectionParser = parser;
        mMapper = new ZTPerspectiveMapper();
        mSnapshots = FastList.newList();

        // Derivative Optimizer
        mOptimizer = new LevenbergMarquardtOptimizer()
            .withCostRelativeTolerance(1.0e-12)
            .withParameterRelativeTolerance(1.0e-12);

        mEstimator = new OptimalParamEstimator();

        setNewOptimizer();
    }

    private BaseOptimizer setNewOptimizer() {
        setPowellOptimizer();
        return mDSMOptimizer;
    }

    private void setPowellOptimizer() {
        mDSMOptimizer = new PowellOptimizer(1.0e-6, 1.0e-6);
    }

    private void setBOBYQAOOptimizer(int numParam) {
        // mDSMOptimizer = new BOBYQAOptimizer(6, 1.0e-1, 1.0e-8);
        // mDSMOptimizer = new BOBYQAOptimizer(20, 1.0e-1, 1.0e-8);

        int numInterpol = (numParam+1)*(numParam+2)/2;
        mDSMOptimizer = new BOBYQAOptimizer(numInterpol, 1.0e-1, 1.0e-8);
    }


    /*
     * Domain Model
     */
    private double computeP(Snapshot s, double[] p) {
        double dx = p[0];
        double dy = p[1];
        double dz = p[2];
        double ka = p[3];
        double kb = p[4];

        Face face = new Face(s.getCameraParams(), new ZTPerspectiveMapper());

        // Set Anchor Correction And Kappa
        INDArray eCorrection = makeAnchorCorrection(dx, dy, dz);
        face.setEyeCenterOfRotationCorrection(eCorrection);
        face.setRightEyeKappa(ka, kb);
        face.setLeftEyeKappa(ka, kb);       // TODO: 부호 반대 처리

        // Update
        face.updateFace(s.getFacePointsNDArr(), s.getIrisCentersInPixelNDArr(), s.getFaceDetailsNDArr());

        // Get P
        INDArray PR = face.getPointOfGazeOfRightEye();

        return distanceL2(s, PR);
    }

    private double computeP(Snapshot s, double dx, double dy, double dz, double ka, double kb) {
       return computeP(s, new double[] {dx, dy, dz, ka, kb});
    }



    /*
     * Calibration
     */
    public double[] calibrate() {
        if (mSnapshots.size() == 0) {
            return null;
        }

        // double[] baseParams = byDSMNearLensKappa();
        double[] p = byDSMDeltaOnly();
        // double[] p = byApproxDSM();
        // double[] p = byDSMDeltaXYOnly();
        // double[] p = byDirectSearchMethod();
        // double[] p = byDerivativeMethod();
        // double[] p = byIterativeDirectSearchMethodDeltaFirst();
        // double[] p = byIterativeDirectSearchMethod();
        // double[] p = byDirectSearchMethodKappa();
        // double[] p = byDSMKappaAndDepthOnly();
        // double[] p = byMeanDSMTwoPass();
        // double[] p = byMeanDirectSearchMethod();
        // double[] p = byDSMLocalRegions();
        // double[] p = byDSMEach();
        // double[] p = byDSMKappaOnly();
        // double[] p = byDSMEachKappaOnly();

        return makeCalibrationResult(p);
    }


    //
    // Approx DSM
    //
    private double[] byApproxDSM() {
        setBOBYQAOOptimizer(6);

        PointValuePair result = (PointValuePair) mDSMOptimizer.optimize(
            GoalType.MINIMIZE,
            new MaxEval(20000),
            new ObjectiveFunction(makeApproxDSM(toRad(3.0), toRad(4.0))),
            new InitialGuess(new double[]{ mInitDx, mInitDy, mInitDz, 0.1, 0.1, 0.1 }),
            new SimpleBounds(getApproxLowerBound(), getApproxUpperBound())
        );

        double[] param = result.getPoint();
        printParam(0, result.getValue(), param);

        return resultWithKappa(param, toRad(3.0), toRad(4.0));
    }

    private MultivariateFunction makeApproxDSM(double ka, double kb) {
        return params -> {
            double se = 0.0;
            for (Snapshot s : mSnapshots) {
                double[] p = computeDeltas(s, params);
                se += computeP(s, p[0], p[1], p[2], ka, kb);
            }

            return se;
        };
    }

    private double[] computeDeltas(Snapshot s, double[] params) {
        double dx = params[0];
        double dy = params[1];
        double dz = params[2];
        double bx = params[3];
        double by = params[4];
        double bz = params[5];

        double depth = s.getE().getDouble(2, 0) / 100.0;

        double x = dx + bx*depth;
        double y = dy + by*depth;
        double z = dz + bz*depth;

        return new double[] { x, y, z };
    }

    private double[] getApproxLowerBound() {
        return new double[] {
                mInitDx + -mDxBound, mInitDy + -mDyBound, mInitDz + -mDzBound, -1.0, -1.0, -1.0
        };
    }

    private double[] getApproxUpperBound() {
        return new double[] {
                mInitDx + mDxBound, mInitDy + mDyBound, mInitDz + mDzBound, 1.0, 1.0, 1.0 };
    }


    /*
     * Optimizer (DSM)
     */
    private double[] byDSMWithParamMap() {
        setNewOptimizer();
        double[] p = getInitGuess();

        // snapshot
        for (int i = 0; i < mSnapshots.size(); i++) {
            PointValuePair result = (PointValuePair) mDSMOptimizer.optimize(
                GoalType.MINIMIZE,
                new MaxEval(20000),
                new ObjectiveFunction(makeDSMObjectiveFunc(i)),
                new InitialGuess(getInitGuess(p))
            );

            p = result.getPoint();
            printParam(i, result.getValue(), p);
            mSnapshots.get(i).setOptimalParam(p);
        }

        // param map
       PointValuePair distResult = (PointValuePair) mDSMOptimizer.optimize(
            GoalType.MINIMIZE,
            new MaxEval(20000),
            new ObjectiveFunction(makeParamObj()),
            new InitialGuess(null)
       );

        double[] distParam = distResult.getPoint();
        Log.e(TAG, String.format(">>> %s", Arrays.toString(distParam)));

        return p;
    }

    private double[] byDSMEach() {
        setBOBYQAOOptimizer(5);

        double[] opt = null;
        double[] initParam = getInitGuess();

        for (int i = 0; i < mSnapshots.size(); i++) {
            PointValuePair result = (PointValuePair) mDSMOptimizer.optimize(
                GoalType.MINIMIZE,
                new MaxEval(20000),
                new ObjectiveFunction(makeDSMObjectiveFunc(i)),
                new InitialGuess(getInitGuess(initParam)),
                new SimpleBounds(getLowerBound(), getUpperBound())
            );

            double error = result.getValue();
            double[] param = result.getPoint();
            printParam(i, error, param);

            initParam = param;
            mSnapshots.get(i).setOptimalParam(param);

            if (i==2 || i == mSnapshots.size()-1) {
                opt = param;
            }
        }

        return opt;
    }

    private double[] byDSMLocalRegions() {
        setNewOptimizer();
        double[] initParam = getInitGuess();

        for (int i=1; i < mSnapshots.size()/2; i++) {
            PointValuePair result = (PointValuePair) mDSMOptimizer.optimize(
                GoalType.MINIMIZE,
                new MaxEval(20000),
                new ObjectiveFunction(makeDSMRegion(new int[]{i, i + 1})),
                new InitialGuess(initParam)
            );

            double error = result.getValue();
            double[] param = result.getPoint();
            printParam(i, error, param);

            initParam = param;
            mSnapshots.get(i).setOptimalParam(param);
        }

        return initParam;
    }

    private double[] byDSMNearLensKappa() {
        mDSMOptimizer = new PowellOptimizer(1.0e-6, 1.0e-6);

        PointValuePair result = (PointValuePair) mDSMOptimizer.optimize(
            GoalType.MINIMIZE,
            new MaxEval(20000),
            new ObjectiveFunction(makeDSMObjectiveFunc(0)),
            new InitialGuess(getInitGuess())
        );

        double[] param = result.getPoint();
        double error = result.getValue();
        Log.e(TAG, String.format(">>> [LENS] err:%f / p:%s", error, Arrays.toString(param)));

        return param;
    }

    private double[] byDSMDeltaOnly() {
        double[] resultParam = null;
        double[] chainParam = getInitDeltaGuess();
        setBOBYQAOOptimizer(3);

        for (int i=0; i<mSnapshots.size(); i++) {
            PointValuePair result = (PointValuePair) mDSMOptimizer.optimize(
                GoalType.MINIMIZE,
                new MaxEval(20000),
                new ObjectiveFunction(makeDSMDeltaOnly(i, toRad(3.0), toRad(4.0))),
                new InitialGuess(chainParam),
                new SimpleBounds(getDeltaLowerBound(), getDeltaUpperBound())
            );

            chainParam = result.getPoint();
            printParam(i, result.getValue(), chainParam);

            if (resultParam == null && (i==mSnapshots.size()/2 || i == mSnapshots.size()-1)) {
                resultParam = chainParam;
            }
        }

        return resultWithKappa(resultParam, toRad(3.0), toRad(4.0));
    }

    private double[] byDSMDeltaXYOnly() {
        double[] resultParam = null;
        double[] chainParam = getInitDeltaXYGuess();
        setBOBYQAOOptimizer(2);

        for (int i=0; i<mSnapshots.size(); i++) {
            PointValuePair result = (PointValuePair) mDSMOptimizer.optimize(
                GoalType.MINIMIZE,
                new MaxEval(2000),
                new ObjectiveFunction(makeDSMDeltaXYOnly(i, mInitDz, toRad(3.0), toRad(4.0))),
                new InitialGuess(chainParam),
                new SimpleBounds(getDeltaXYLowerBound(), getDeltaXYUpperBound())
            );

            chainParam = result.getPoint();
            printParam(i, result.getValue(), chainParam);

            if (resultParam == null && (i==mSnapshots.size()/2 || i == mSnapshots.size()-1)) {
                resultParam = chainParam;
            }
        }

        return resultWithKappa(resultParam, mInitDz, toRad(3.0), toRad(4.0));
    }

    private double[] byMeanDirectSearchMethod() {
        double[] meanParams = new double[5];
        mDSMOptimizer = new PowellOptimizer(1.0e-6, 1.0e-6);

        int numSamples = 0;
        for (int i=0; i<mSnapshots.size(); i++) {
            PointValuePair result = (PointValuePair) mDSMOptimizer.optimize(
                GoalType.MINIMIZE,
                new MaxEval(20000),
                new ObjectiveFunction(makeDSMObjectiveFunc(i)),
                new InitialGuess(getInitGuess())
            );

            numSamples += 1;
            double[] param = result.getPoint();
            double error = result.getValue();

            Log.e(TAG, String.format(">>> [%3d] err:%f / p:%s", i, error, Arrays.toString(param)));
            // Log.e(TAG, String.format(">>> %s", mSnapshots.get(i)));

            for (int i1 = 0; i1 < param.length; i1++) {
                meanParams[i1] += param[i1];
            }
        }

        for (int i=0; i<meanParams.length; i++) {
            meanParams[i] /= numSamples;
        }

        return meanParams;
    }

    private double[] byMeanDSMTwoPass() {
        double[] mMeanParams = new double[5];
        mDSMOptimizer = new PowellOptimizer(1.0e-6, 1.0e-6);

        int numSamples = 0;
        for (int i=0; i<mSnapshots.size(); i++) {
            PointValuePair result = (PointValuePair) mDSMOptimizer.optimize(
                GoalType.MINIMIZE,
                new MaxEval(20000),
                new ObjectiveFunction(makeDSMObjectiveFunc(i)),
                new InitialGuess(getInitGuess())
            );

            numSamples += 1;

            double[] point = result.getPoint();
            double error = result.getValue();

            Log.e(TAG, String.format("*** [%d] err:[%d] P:%s ***", i, ((int) error), Arrays.toString(point)));
            for (int i1 = 0; i1 < point.length; i1++) {
                mMeanParams[i1] += point[i1];
            }
        }

        mDSMOptimizer = new PowellOptimizer(1.0e-6, 1.0e-6);
        for (int i=0; i<mMeanParams.length; i++) {
            mMeanParams[i] /= numSamples;
        }

        double[] meanParam = mMeanParams;

        PointValuePair result = (PointValuePair) mDSMOptimizer.optimize(
                GoalType.MINIMIZE,
                new MaxEval(20000),
                new ObjectiveFunction(makeDSMKappaObjectiveFunc(
                    new double[] { meanParam[0], meanParam[1], meanParam[2] })),
                new InitialGuess(new double[] { meanParam[3], meanParam[4] })
            );

        double[] param = result.getPoint();
        double error = result.getValue();

        Log.e(TAG, String.format("*** DSM Err:[%d]Param:%s ***", ((int) error), Arrays.toString(param)));
        return new double[] {meanParam[0], meanParam[1], meanParam[2], param[0], param[1] };

    }

    private double[] byDSMKappaOnly() {
        setBOBYQAOOptimizer(2);

        PointValuePair result = (PointValuePair) mDSMOptimizer.optimize(
            GoalType.MINIMIZE,
            new MaxEval(200000),
            new ObjectiveFunction(makeKappaOnly()),
            new InitialGuess(new double[] { mInitKappaAlpha, mInitKappaBeta }),
            new SimpleBounds(getKappaLowerBound(), getKappaUpperBound())
        );

        double[] param = result.getPoint();
        double error = result.getValue();
        Log.e(TAG, String.format(">>> %.3f | [%.1f, %.1f]", error, param[0], param[1]));

        return new double[] { mInitDx, mInitDy, mInitDz, param[0], param[1]};
    }

    private double[] byDSMEachKappaOnly() {
        setBOBYQAOOptimizer(2);

        double[] p = null;
        for (int i = 0; i < mSnapshots.size(); i++) {
            PointValuePair result = (PointValuePair) mDSMOptimizer.optimize(
                GoalType.MINIMIZE,
                new MaxEval(20000),
                new ObjectiveFunction(makeKappaOnly(i)),
                new InitialGuess(new double[] { mInitKappaAlpha, mInitKappaBeta }),
                new SimpleBounds(getKappaLowerBound(), getKappaUpperBound())
            );

            double error = result.getValue();
            p = result.getPoint();
            printParam(i, result.getValue(), new double[] {mInitDx, mInitDy, mInitDz, p[0], p[1]});
        }

        return new double[] { mInitDx, mInitDy, mInitDz, p[0], p[1]};
    }

    private double[] byDSMKappaAndDepthOnly() {
        mDSMOptimizer = new PowellOptimizer(1.0e-6, 1.0e-6);

        PointValuePair result = (PointValuePair) mDSMOptimizer.optimize(
                GoalType.MINIMIZE,
                new MaxEval(200000),
                new ObjectiveFunction(makeDepthAndKappaOnly()),
                new InitialGuess(new double[] { mInitDz, mInitKappaAlpha, mInitKappaBeta })
            );

        double[] param = result.getPoint();
        double error = result.getValue();

        Log.e(TAG, String.format("*** DSM Err:[%d]Param:%s ***", ((int) error), Arrays.toString(param)));
        return new double[] { mInitDx, mInitDy, param[0], param[1], param[2]};
    }

    private double[] byDirectSearchMethod() {
        setBOBYQAOOptimizer(5);

        PointValuePair result = (PointValuePair) mDSMOptimizer.optimize(
                GoalType.MINIMIZE,
                new MaxEval(200000),
                new ObjectiveFunction(makeDSMObjectiveFuncAll()),
                new InitialGuess(getInitGuess()),
                new SimpleBounds(getLowerBound(), getUpperBound())
            );

        double[] param = result.getPoint();
        double error = result.getValue();

        Log.e(TAG, String.format("*** err:%d p:%s ***", ((int) error), Arrays.toString(param)));
        return param;
    }

    private double[] byIterativeDirectSearchMethod() {
        setBOBYQAOOptimizer(2);
        PointValuePair result = (PointValuePair) mDSMOptimizer.optimize(
                GoalType.MINIMIZE,
                new MaxEval(20000),
                new ObjectiveFunction(makeDSMKappaObjectiveFunc(getInitDeltaGuess())),
                new InitialGuess(getInitKappaGuess()),
                new SimpleBounds(getKappaLowerBound(), getKappaUpperBound())
            );

        double[] kappaParam = result.getPoint();
        double kappaError = result.getValue();

        Log.e(TAG, String.format("*** err:%d ks: %s ***", ((int) kappaError), Arrays.toString(kappaParam)));

        setBOBYQAOOptimizer(3);
        result = (PointValuePair) mDSMOptimizer.optimize(
                GoalType.MINIMIZE,
                new MaxEval(20000),
                new ObjectiveFunction(makeDSMDeltaObjectiveFunc(kappaParam)),
                new InitialGuess(getInitDeltaGuess()),
                new SimpleBounds(getDeltaLowerBound(), getDeltaUpperBound())
            );

        double[] deltaParam = result.getPoint();
        double deltaError = result.getValue();

        Log.e(TAG, String.format("*** err:%d deltas: %s ***", ((int) deltaError), Arrays.toString(deltaParam)));


        return ArrayUtils.addAll(deltaParam, kappaParam);
    }

    private double[] byIterativeDirectSearchMethodDeltaFirst() {

        PowellOptimizer deltaOptimizer = new PowellOptimizer(1.0e-1, 1.0e-5);
        PointValuePair result = deltaOptimizer.optimize(
                GoalType.MINIMIZE,
                new MaxEval(20000),
                new ObjectiveFunction(makeDSMDeltaObjectiveFunc(getInitKappaGuess())),
                new InitialGuess(getInitDeltaGuess())
            );

        double[] deltaParam = result.getPoint();
        double deltaError = result.getValue();

        Log.e(TAG, String.format("*** DSM Err:[%d] Delta :%s ***", ((int) deltaError), Arrays.toString(deltaParam)));


        PowellOptimizer kappaOptimizer = new PowellOptimizer(1.0e-10, 1.0e-10);
        result = kappaOptimizer.optimize(
                GoalType.MINIMIZE,
                new MaxEval(20000),
                new ObjectiveFunction(makeDSMKappaObjectiveFunc(deltaParam)),
                new InitialGuess(getInitKappaGuess())
            );

        double[] kappaParam = result.getPoint();
        double kappaError = result.getValue();

        Log.e(TAG, String.format("*** DSM Err:[%d] kappa:%s ***", ((int) kappaError), Arrays.toString(kappaParam)));

        return ArrayUtils.addAll(deltaParam, kappaParam);
    }

    private double[] byDirectSearchMethodKappa() {
        PowellOptimizer kappaOptimizer = new PowellOptimizer(1.0e-10, 1.0e-10);

        PointValuePair result = kappaOptimizer.optimize(
                GoalType.MINIMIZE,
                new MaxEval(200000),
                new ObjectiveFunction(makeDSMKappaObjectiveFunc(getInitDeltaGuess())),
                new InitialGuess(getInitKappaGuess())
            );

        double[] kappaParam = result.getPoint();
        double kappaError = result.getValue();

        Log.e(TAG, String.format("*** DSM Err:[%d] Delta :%s ***", ((int) kappaError), Arrays.toString(kappaParam)));

        return ArrayUtils.addAll(getInitDeltaGuess(), kappaParam);
    }


    private MultivariateFunction makeParamObj() {
        return params -> {

            double se = 0.0;
            for (Snapshot s : mSnapshots) {

            }

            return se;
        };
    }

    private MultivariateFunction makeDSMDeltaOnly(int snapshotIdx, double ka, double kb) {
        return params -> {
            double dx = params[0];
            double dy = params[1];
            double dz = params[2];
            Snapshot s = mSnapshots.get(snapshotIdx);
            double err = computeP(s, dx, dy, dz, ka, kb);
            return err;
        };
    }

    private MultivariateFunction makeDSMDeltaXYOnly(int snapshotIdx, double dz, double ka, double kb) {
        return params -> {
            double dx = params[0];
            double dy = params[1];
            Snapshot s = mSnapshots.get(snapshotIdx);
            double err = computeP(s, dx, dy, dz, ka, kb);
            return err;
        };
    }

    private MultivariateFunction makeDSMObjectiveFunc(int snapshotIdx) {
        return params -> {
            double dx = params[0];
            double dy = params[1];
            double dz = mUsingDz ? params[2] : mInitDz;
            double ka = mUsingKappa ? params[3] : 0.0;
            double kb = mUsingKappa ? params[4] : 0.0;

            Snapshot s = mSnapshots.get(snapshotIdx);
            double err = computeP(s, dx, dy, dz, ka, kb);
            // Log.e(TAG, String.format("*** [%.3f %.3f %.3f %.2f %.2f] %3.1f ***", dx, dy, dz, ka, kb, err));
            return err;
        };
    }

    private MultivariateFunction makeDSMRegion(int[] ids) {
        return params -> {
            double dx = params[0];
            double dy = params[1];
            double dz = params[2];
            double ka = params[3];
            double kb = params[4];

            double se = 0.0;
            for (int id : ids) {
                se += computeP(mSnapshots.get(id), dx, dy, dz, ka, kb);
            }

            Log.e(TAG, String.format("*** [%.1f %.1f %.1f %.6f %.6f] %3.1f ***", dx, dy, dz, ka, kb, se));
            return se;
        };
    }

    private MultivariateFunction makeDSMObjectiveFuncAll() {
        return params -> {
            double dx = params[0];
            double dy = params[1];
            double dz = params[2];
            double ka = params[3];
            double kb = params[4];

            double se = 0.0;
            for (Snapshot s : mSnapshots) {
                se += computeP(s, dx, dy, dz, ka, kb);
            }

            Log.e(TAG, String.format("*** [%.1f %.1f %.1f %.6f %.6f] %3.1f ***", dx, dy, dz, ka, kb, se));
            return se;
        };
    }

    private MultivariateFunction makeDepthAndKappaOnly() {
        return params -> {
            double dz = params[0];
            double ka = params[1];
            double kb = params[2];

            double se = 0.0;
            for (Snapshot s : mSnapshots) {
                se += computeP(s, mInitDx, mInitDy, dz, ka, kb);
            }

            Log.e(TAG, String.format("*** [%.1f %.1f %.1f %.6f %.6f] %3.1f ***", mInitDx, mInitDy, dz, ka, kb, se));
            return se;
        };
    }

    private MultivariateFunction makeKappaOnly() {
        return params -> {
            double ka = params[0];
            double kb = params[1];

            double se = 0.0;
            for (Snapshot s : mSnapshots) {
                se += computeP(s, mInitDx, mInitDy, mInitDz, ka, kb);
            }

            return se;
        };
    }

    private MultivariateFunction makeKappaOnly(int idx) {
        return params -> {
            double ka = params[0];
            double kb = params[1];
            double se = computeP(mSnapshots.get(idx), mInitDx, mInitDy, mInitDz, ka, kb);
            return se;
        };
    }

    private MultivariateFunction makeDSMDeltaObjectiveFunc(double[] kappaParam) {
        double ka = kappaParam[0];
        double kb = kappaParam[0];

        return params -> {
            double dx = params[0];
            double dy = params[1];
            double dz = params[2];

            double se = 0.0;
            for (Snapshot s : mSnapshots) {
                se += computeP(s, dx, dy, dz, ka, kb);
            }

            return se;
        };
    }

    private MultivariateFunction makeDSMKappaObjectiveFunc(double[] deltaParam) {
        double dx = deltaParam[0];
        double dy = deltaParam[1];
        double dz = deltaParam[2];

        return params -> {
            double ka = params[0];
            double kb = params[1];

            double se = 0.0;
            for (Snapshot s : mSnapshots) {
                se += computeP(s, dx, dy, dz, ka, kb);
            }

            return se;
        };
    }

    private double[] getInitGuess() {
        return new double[]{mInitDx, mInitDy, mInitDz, mInitKappaAlpha, mInitKappaBeta};
    }

    private double[] getInitGuess(double[] p) {
        if (mUsingKappa) {
            return new double[] { p[0], p[1], p[2], p[3], p[4] };
        } else if (mUsingDz) {
            return new double[] { p[0], p[1], p[2], 0, 0 };
        } else {
            return new double[] { p[0], p[1], mInitDz, 0, 0 };
        }
    }

    private double[] getInitDeltaGuess() {
        return new double[] { mInitDx, mInitDy, mInitDz };
    }

    private double[] getInitDeltaXYGuess() {
        return new double[] { mInitDx, mInitDy };
    }

    private double[] getInitKappaGuess() {
        return new double[] { mInitKappaAlpha, mInitKappaBeta };
    }

    private double[] getLowerBound() {
        return new double[] {
                mInitDx + -mDxBound,
                mInitDy + -mDyBound,
                mInitDz + -mDzBound,
                -mKappaAlphaBound,
                -mKappaBetaBound
        };
    }

    private double[] getUpperBound() {
        return new double[] {
                mInitDx + mDxBound,
                mInitDy + mDyBound,
                mInitDz + 0,
                mKappaAlphaBound,
                mKappaBetaBound
        };
    }

    private double[] getDeltaLowerBound() {
        return new double[] {
                mInitDx + -mDxBound,
                mInitDy + -mDyBound,
                mInitDz + -mDzBound,
        };
    }

    private double[] getDeltaUpperBound() {
        return new double[] {
                mInitDx + mDxBound,
                mInitDy + mDyBound,
                mInitDz + mDzBound,
        };
    }

    private double[] getDeltaXYLowerBound() {
        return new double[] { mInitDx + -mDxBound, mInitDy + -mDyBound, };
    }

    private double[] getDeltaXYUpperBound() {
        return new double[] { mInitDx + mDxBound, mInitDy + mDyBound, };
    }

    private double[] getKappaLowerBound() {
        return new double[] { -mKappaAlphaBound, -mKappaBetaBound};
    }

    private double[] getKappaUpperBound() {
        return new double[] { mKappaAlphaBound, mKappaBetaBound};
    }



    /*
     * Distance Factor Calibration
     */
    public double calibrateDistFactor(double[] params) {
        // return byDSMDistFactorAll(params);
        return byDSMDistFactorEach(params);
    }

    private double byDSMDistFactorAll(double[] params) {
        mDSMOptimizer = new PowellOptimizer(1.0e-6, 1.0e-6);

        PointValuePair result = (PointValuePair) mDSMOptimizer.optimize(
                GoalType.MINIMIZE,
                new MaxEval(2000),
                new ObjectiveFunction(byDSMDistanceFactor(params)),
                new InitialGuess(new double[] { 1.0 })
            );

        double[] param = result.getPoint();
        double error = result.getValue();
        Log.e(TAG, String.format("*** DSM Dist Err:[%d] Param:%s ***", ((int) error), Arrays.toString(param)));

        return param[0];
    }

    private double byDSMDistFactorEach(double[] params) {
        mDSMOptimizer = new PowellOptimizer(1.0e-6, 1.0e-6);

        for (int i=0; i<mSnapshots.size(); i++) {
            PointValuePair result = (PointValuePair) mDSMOptimizer.optimize(
                GoalType.MINIMIZE,
                new MaxEval(2000),
                new ObjectiveFunction(byDSMDistanceEach(i, params)),
                new InitialGuess(new double[]{1.0})
            );

            double err = result.getValue();
            double[] factor = result.getPoint();
            Log.e(TAG, String.format("*** [%d] %.3f %3.1f ***", i, factor[0], err));
        }

        return 1.0;
    }

    private MultivariateFunction byDSMDistanceFactor(double[] deltaAndKappa) {
        return params -> {
            double se = 0.0;
            double distFactor = params[0];
            for (Snapshot s : mSnapshots) {
                se += computeP(s, deltaAndKappa);
            }

            Log.e(TAG, String.format("*** %.1f %3.1f ***", distFactor, se));
            return se;
        };
    }

    private MultivariateFunction byDSMDistanceEach(int idx, double[] deltaAndKappa) {
        return params -> {
            double err = computeP(mSnapshots.get(idx), deltaAndKappa);
            return err;
        };
    }



    /*
     * Optimizer (Derivative)
     */
    private double[] byDerivativeMethod() {
        LeastSquaresProblem cp = makeCalibrationProblem();
        LeastSquaresOptimizer.Optimum optimum = mOptimizer.optimize(cp);

        if (optimum == null) {
            return null;
        }

        return optimum.getPoint().toArray();
    }

    private LeastSquaresProblem makeCalibrationProblem() {
        LeastSquaresBuilder pb = new LeastSquaresBuilder()
            .start(new double[] { 0, 0, 0, 0, 0 })              // kappa 평균 각도에서 시작할 것
            .model(makePoGModel())
            .target(new double[mSnapshots.size()])
            // .parameterValidator(makeBoundConditions(mDtBound, mKappaAlphaBound, mKappaBetaBound))
            .maxEvaluations(1000)
            .maxIterations(1000);

        return pb.build();
    }

    private LeastSquaresConverter applyConverter(MultivariateVectorFunction f) {
        double[] observations = new double[mSnapshots.size()];  // All Zeros
        LeastSquaresConverter c = new LeastSquaresConverter(f, observations);
        return c;
    }

    private MultivariateJacobianFunction makePoGModel() {
        return params -> {
            double dx = params.getEntry(0);
            double dy = params.getEntry(1);
            double dz = params.getEntry(2);
            double ka = params.getEntry(3);
            double kb = params.getEntry(4);

            int num_samples = mSnapshots.size();
            RealVector errors = new ArrayRealVector(num_samples);
            RealMatrix jacobian = new Array2DRowRealMatrix(num_samples, 5);

            for (int i=0; i<num_samples; i++) {
                Snapshot ss = mSnapshots.get(i);

                // errors
                double err = computeP(ss, dx, dy, dz, ka, kb);
                Log.e(TAG, String.format("*** POG[%d] ERR: %f when %f %f %f %f %f ***", i, err, dx, dy, dz, ka, kb));
                errors.setEntry(i, err);

                // jacobian
                jacobian.setEntry(i, 0, err);       // dx
                jacobian.setEntry(i, 1, err);       // dy
                jacobian.setEntry(i, 2, err);       // dz
                jacobian.setEntry(i, 3, err);       // ka
                jacobian.setEntry(i, 4, err);       // kb
            }

            return new Pair<>(errors, jacobian);
        };
    }

    private ParameterValidator makeBoundConditions(double dxBound, double alphaBound, double betaBound) {
        return params -> null;
    }



    /*
     * Helper
     */
    private double[] resultWithKappa(double[] p, double ka, double kb) {
        return new double[] { p[0], p[1], p[2], ka, kb };
    }

    private double[] resultWithKappa(double[] p, double dz, double ka, double kb) {
        return new double[] { p[0], p[1], dz, ka, kb };
    }

    private double[] makeCalibrationResult(double[] p) {
        double[] result;

        if (mUsingKappa) {
            result = new double[] { p[0], p[1], p[2], p[3], p[4] };
        } else if (mUsingDz) {
            result = new double[] { p[0], p[1], p[2], 0, 0 };
        } else {
            result = new double[] { p[0], p[1], mInitDz, 0, 0 };
        }

        Log.e(TAG, String.format(">>> Final deltas: %.2f, %.2f, %.2f Kappa degree: %.2f %.2f",
                p[0], p[1], p[2], toDeg(p[3]), toDeg(p[4])));

        return result;
    }

    private void printParam(int i, double err, double[] p) {
        // Only Delta XY
        if (p.length == 2) {
            Log.e(TAG, String.format(">>> [%3d:%.3f] / [%.2f, %.2f, %.2f]", i, err, p[0], p[1], mInitDz));
            return;
        }

        // Only Delta
        if (p.length == 3) {
            Log.e(TAG, String.format(">>> [%3d:%.3f] / [%.2f, %.2f, %.2f]", i, err, p[0], p[1], p[2]));
            return;
        }

        double dx = p[0];
        double dy = p[1];
        double dz = mUsingDz ? p[2] : mInitDz;
        double ka = mUsingKappa ? p[3] : 0.0;
        double kb = mUsingKappa ? p[4] : 0.0;

        if (mUsingKappa) {
            Log.e(TAG, String.format(">>> [%3d:%.3f] / [%.2f, %.2f, %.2f, %.1f, %.1f]", i, err, dx, dy, dz, ka, kb));
        } else if (mUsingDz) {
            Log.e(TAG, String.format(">>> [%3d:%.3f] / [%.2f, %.2f, %.2f]", i, err, dx, dy, dz));
        } else {
            Log.e(TAG, String.format(">>> [%3d:%.3f] / [%.2f, %.2f]", i, err, dx, dy));
        }
    }

    private void learnParam() {
        mEstimator.fit(mSnapshots.subList(0, mSnapshots.size()-2));
    }

    private void printItem(INDArray E, INDArray I, INDArray R) {
        Log.e(TAG,
                toStringColVec(E) + "  " +
                toStringColVec(I) + "  " +
                toStringColVec(I.sub(E)) + "  " +
                toStringMat3x3(R)
        );
    }

    private double computeErrorInAllSnapshot(double dx, double dy, double dz, double ka, double kb) {
        double accErr = 0.0;
        for (Snapshot s : mSnapshots) {
            accErr += computeP(s, dx, dy, dz, ka, kb);
        }
        return accErr;
    }


    public void add(Snapshot ss) {
        mSnapshots.add(ss);
    }

    public void preprocess() {
        INDArray eCorrection = makeAnchorCorrection(mInitDx, mInitDy, mInitDz);
        INDArray carunclePosInWorld = mMapper.getEyeCarunclesPointsOfModel();

        for (Snapshot s : mSnapshots) {
            INDArray caruncles = mMapper.carunclesFrom(s.getFaceDetailsNDArr());
            Point[] imagePoints = mMapper.imagePointsFrom(s.getFacePointsNDArr(), s.getFaceDetailsNDArr());
            Point3[] modelPoints = mMapper.get3dModelPoints();

            computeFacePosition(imagePoints, modelPoints, s.getCameraParams(), mRotVec, mTransVec);
            s.getCameraParams().setExtrinsicParam(toExtrinsicMat(mRotVec, mTransVec));

            INDArray E = computeEyeCenterOfRotationByCaruncle(
                    s.getCameraParams(), carunclePosInWorld, caruncles, eCorrection);
            s.setInitialE(E);
        }
    }

}
