package cn.edu.nju.ws.edgqa.eval;

import java.text.DecimalFormat;

public class CumulativeIRMetrics {
    private static DecimalFormat decimalFormat = new DecimalFormat("0.000");
    private double cumulativePrecision = 0.0;
    private double cumulativeRecall = 0.0;
    private double cumulativeMicroF1 = 0.0;
    private int numSample = 0;

    public CumulativeIRMetrics() {
        cumulativePrecision = 0.0;
        cumulativeRecall = 0.0;
        cumulativeMicroF1 = 0.0;
        numSample = 0;
    }

    public CumulativeIRMetrics(double cumulativePrecision, double cumulativeRecall, double cumulativeMicroF1, int numSample) {
        this.cumulativePrecision = cumulativePrecision;
        this.cumulativeRecall = cumulativeRecall;
        this.cumulativeMicroF1 = cumulativeMicroF1;
        this.numSample = numSample;
    }

    public void addSample(double precision, double recall) {
        cumulativePrecision += precision;
        cumulativeRecall += recall;
        cumulativeMicroF1 += (2.0 * precision * recall) / (precision + recall);
        numSample++;
    }

    public void addSample(IRMetrics IRMetrics) {
        cumulativePrecision += IRMetrics.getPrecision();
        cumulativeRecall += IRMetrics.getRecall();
        cumulativeMicroF1 += IRMetrics.getMicroF1();
        numSample++;
    }

    public void addSample(double precision, double recall, double microF1) {
        cumulativePrecision += precision;
        cumulativeRecall += recall;
        cumulativeMicroF1 += microF1;
        numSample++;
    }

    public double getPrecision() {
        if (numSample == 0)
            return 0.0;
        return cumulativePrecision / numSample;
    }

    public double getRecall() {
        if (numSample == 0)
            return 0.0;
        return cumulativeRecall / numSample;
    }

    public double getAverageF1() {
        if (numSample == 0)
            return 0.0;
        return cumulativeMicroF1 / numSample;
    }

    public double getMacroF1() {
        if (getPrecision() + getRecall() == 0.0)
            return 0.0;
        return 2.0 * (getPrecision() * getRecall()) / (getPrecision() + getRecall());
    }

    public String toString() {
        return "sample: " + numSample + ", P: " + decimalFormat.format(getPrecision()) + ", R: " + decimalFormat.format(getRecall()) + ", macro F1: " + decimalFormat.format(getMacroF1());
    }

    public String toQALDString() {
        return "sample: " + numSample + ", P: " + decimalFormat.format(getPrecision()) + ", R: " + decimalFormat.format(getRecall()) + ", average F1: " + decimalFormat.format(getAverageF1()) + ", QALD macro F1: " + decimalFormat.format(getMacroF1());
    }
}
