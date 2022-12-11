package cn.edu.nju.ws.edgqa.utils;

public class Timer {
    /**
     * EDG generation timer
     */
    public static long totalEDGGenTime = 0;
    /**
     * Entity linking timer
     */
    public static long totalEntityLinkingTime = 0;
    /**
     * Entity linking tool timer
     */
    public static long totalEntityLinkingToolTime = 0;
    /**
     * Relation linking timer
     */
    public static long totalRelationLinkingTime = 0;
    /**
     * Similarity calculation timer
     */
    public static long totalSimilarityTime = 0;

    public static long getTotalEntityLinkingToolTime() {
        return totalEntityLinkingToolTime;
    }

    public static void setTotalEntityLinkingToolTime(long totalEntityLinkingToolTime) {
        Timer.totalEntityLinkingToolTime = totalEntityLinkingToolTime;
    }

    public static long getTotalSimilarityTime() {
        return totalSimilarityTime;
    }

    public static void setTotalSimilarityTime(long totalSimilarityTime) {
        Timer.totalSimilarityTime = totalSimilarityTime;
    }

    public static long getTotalEntityLinkingTime() {
        return totalEntityLinkingTime;
    }

    public static void setTotalEntityLinkingTime(long totalEntityLinkingTime) {
        Timer.totalEntityLinkingTime = totalEntityLinkingTime;
    }

    public static long getTotalRelationLinkingTime() {
        return totalRelationLinkingTime;
    }

    public static void setTotalRelationLinkingTime(long totalRelationLinkingTime) {
        Timer.totalRelationLinkingTime = totalRelationLinkingTime;
    }
}
