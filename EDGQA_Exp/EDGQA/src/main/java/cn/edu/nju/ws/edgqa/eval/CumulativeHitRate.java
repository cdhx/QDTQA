package cn.edu.nju.ws.edgqa.eval;

import java.text.DecimalFormat;

public class CumulativeHitRate {
    private static DecimalFormat decimalFormat = new DecimalFormat("0.000");
    private int hit;
    private int numGoldenSample = 0;

    public void addSample(boolean hit) {
        if (hit)
            this.hit++;
        numGoldenSample++;
    }

    public int getNumGoldenSample() {
        return numGoldenSample;
    }

    public double getHitRate() {
        return (double) hit / numGoldenSample;
    }

    public String toString() {
        return "sample: " + numGoldenSample + ", HR: " + decimalFormat.format(getHitRate());
    }
}
