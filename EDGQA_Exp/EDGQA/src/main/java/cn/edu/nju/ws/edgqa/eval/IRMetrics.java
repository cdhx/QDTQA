package cn.edu.nju.ws.edgqa.eval;

public class IRMetrics {
    private static final double eps = 0.000001;
    private double precision;
    private double recall;
    private double microF1;
    private boolean initialized;

    public IRMetrics() {
        precision = 0.0;
        recall = 0.0;
        microF1 = 0.0;
        initialized = false;
    }

    public IRMetrics(double precision, double recall) {
        initialized = true;
        this.precision = precision;
        this.recall = recall;
        this.microF1 = getF1(precision, recall);
    }

    public IRMetrics(double precision, double recall, double F1) {
        initialized = true;
        this.precision = precision;
        this.recall = recall;
        this.microF1 = F1;
    }

    public static double getF1(double precision, double recall) {
        if (precision < eps && recall < eps)
            return 0.0;
        return 2.0 * precision * recall / (precision + recall);
    }

    public double getPrecision() {
        return precision;
    }


    public double getRecall() {
        return recall;
    }


    public double getMicroF1() {
        return microF1;
    }

    public boolean isInitialized() {
        return initialized;
    }

    public void setMetrics(double precision, double recall) {
        initialized = true;
        this.precision = precision;
        this.recall = recall;
        this.microF1 = getF1(precision, recall);
    }

    public void setMetrics(double precision, double recall, double microF1) {
        initialized = true;
        this.precision = precision;
        this.recall = recall;
        this.microF1 = microF1;
    }

    public String toString() {
        return "P: " + precision + ", R: " + recall + ", F: " + microF1;
    }
}
