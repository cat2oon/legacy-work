package camp.visual.kappa.utils;

import android.graphics.PointF;

public class STLine{
    PointF startPoint;
    PointF endPoint;

    float startPointVisiable;
    float endPointVisiable;

        public STLine(PointF p1, PointF p2, float v1, float v2) {
            this.startPoint = p1;
            this.endPoint = p2;

            this.startPointVisiable = v1;
            this.endPointVisiable = v2;
        }
    }