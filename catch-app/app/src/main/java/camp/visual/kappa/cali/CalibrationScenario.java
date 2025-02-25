package camp.visual.kappa.cali;

import android.util.Pair;

import org.eclipse.collections.impl.list.mutable.FastList;


/*
 * 렌즈 기준 위치 mm
 */
public class CalibrationScenario {

    // TODO: Snapshot 으로 가지고 있기
    private FastList<Pair<Double, Double>> mCaliScenario;

    public CalibrationScenario() {
        mCaliScenario = FastList.newList();

        // setNinePointScenario();
        // setSamePointScenario(15.0, -130.0, 1);

        // setFivePointScenario();
        // setGridScenario();
        // setLocalGridScenario();

        setSamePointScenario(-15.0, -70.0, 9);
        // setSamePointScenario(15.0, -10.0, 1);
        // setSamePointScenario(0, -10.0, 1);
    }



    //
    // Accessor & Mutator
    //
    public void addScenario(double mxFromLens, double myFromLens) {
        mCaliScenario.add(Pair.create(mxFromLens, myFromLens));
    }

    public int getNumScenario() {
        return mCaliScenario.size();
    }

    public Pair<Double, Double> getCurrentTarget(int idx) {
        return mCaliScenario.get(idx);
    }



    //
    // Preset Scenario
    //
    private void setFirstScenarioAsNearLens() {
        mCaliScenario.add(0, Pair.create(0.0, -8.0));
    }

    private void setFivePointScenario() {
        addScenario(-40.0,  -10.0);
        addScenario( 15.0,  -10.0);
        addScenario(-15.0,  -70.0);
        addScenario(-40.0, -130.0);
        addScenario( 15.0, -130.0);
    }

    private void setNinePointScenario() {
        addScenario(-40.0,   -10.0); // 1
        addScenario(-15.0,   -10.0); // 2
        addScenario( 15.0,   -10.0); // 3

        addScenario(-40.0,   -70.0); // 4
        addScenario(-15.0,   -70.0); // 5
        addScenario( 15.0,   -70.0); // 6

        addScenario(-40.0,  -130.0); // 7
        addScenario(-15.0,  -130.0); // 8
        addScenario( 15.0,  -130.0); // 9
    }

    private void setGridScenario() {
        for (int i=-40; i<=15; i+=10) {
            for (int j=-20; j>=-120; j-=20) {
                addScenario(i, j);
            }
        }
    }

    private void setLocalGridScenario() {
        double[][] regions = new double[][] {
            new double[] {-40.0,  -10.0},
            new double[] { 15.0,  -10.0},
            new double[] {-15.0,  -70.0},
            new double[] {-40.0, -130.0},
            new double[] { 15.0, -130.0},
        };

        for (int i=0; i<regions.length; i++) {
            double x = regions[i][0];
            double y = regions[i][1];

            addScenario(x, y);
            addScenario(x+1, y-1);
        }
    }

    private void setSamePointScenario(double x, double y) {
        // addScenario(-15.0,  -70.0);
       setSamePointScenario(x, y, 3);
    }

    private void setSamePointScenario(double x, double y, int count) {
        for (int i=0; i<count; i++) {
            addScenario(x, y);
        }
    }

}
