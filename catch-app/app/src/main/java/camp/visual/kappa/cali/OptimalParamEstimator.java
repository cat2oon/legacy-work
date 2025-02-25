package camp.visual.kappa.cali;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.eclipse.collections.api.list.MutableList;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * 영향을 줄 수 있는 요소 후보 정리
 *
 * # 확정 요소
 * - Head Pose (R mat) (9)
 * - E Before Correction (렌즈와 안구의 거리 (x, y, z)) (3)
 * - IrisPixel || 안축 벡터 (OPTx, OPTy, OPTz) (2|3)
 *
 * # 후보 요소
 * - Kappa
 * - 캘리브레이션 당시 파라미터
 * - 해상도
 * - 렌즈 종류 (focal length, fov, 렌즈 퀄리티, 방사 왜곡률, distortion coefficient)
 *
 *
 * @ EXP.01
 * - f(R, E, IrisPixel) = pred(dx, dy)
 *
 */
public class OptimalParamEstimator {

    private int mNumLabel = 2;          // { dx, dy, dz, ka, kb }
    private int mNumInputs = 14;        // { E, R, IRIS PIXEL }
    private int mNumEpochs = 5;
    private MultiLayerNetwork mNN;


    public OptimalParamEstimator() {
        mNN = makeNeuralNet();
        mNN.init();
    }

    private MultiLayerNetwork makeNeuralNet() {

        MultiLayerConfiguration nnConf = new NeuralNetConfiguration.Builder()
            .seed(860515)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(new Adam())
            .l2(1e-4)
            .list()
            .layer(new DenseLayer.Builder()
                .nIn(mNumInputs)
                .nOut(mNumInputs * 32)
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .build()
            )
            .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                .nIn(mNumInputs * 32)
                .nOut(mNumLabel)
                .weightInit(WeightInit.XAVIER)
                .build()
            )
            .pretrain(false)
            .backprop(true)
            .build();

        return new MultiLayerNetwork(nnConf);
    }


    //
    //
    //
    public void fit(MutableList<Snapshot> snapshots) {
        INDArray inputs = toInputs(snapshots);
        INDArray labels = toLabels(snapshots);

        for (int e=0; e<mNumEpochs; e++) {
            mNN.fit(inputs, labels);
        }
    }

    public INDArray pred(Snapshot s) {
        return mNN.output(toInput(s));
    }


    //
    // Snapshot to dataset
    //
    private INDArray toLabels(MutableList<Snapshot> snapshots) {
        double[] arr = new double[snapshots.size() * mNumLabel];

        for (int i=0; i<snapshots.size(); i++) {
            Snapshot s = snapshots.get(i);
            double[] p = s.getOptimalParam();

            for (int j=0; j<mNumLabel; j++) {
                arr[i + j] = p[j];
            }
        }

        return Nd4j.create(arr, new int[]{snapshots.size(), mNumLabel});
    }

    private INDArray toInputs(MutableList<Snapshot> snapshots) {
        INDArray inputs = null;
        for (int i=0; i<snapshots.size(); i++) {
            INDArray item = toInput(snapshots.get(i));
            inputs = inputs == null ? item : Nd4j.vstack(inputs, item);
        }

        return inputs;
    }

    private INDArray toInput(Snapshot s) {
        INDArray r = s.getRotationMat();
        INDArray i = s.getIrisCentersInPixelNDArr().getColumn(0);
        return Nd4j.hstack(r.getRow(0), r.getRow(1), r.getRow(2), i.transpose());
    }

}
