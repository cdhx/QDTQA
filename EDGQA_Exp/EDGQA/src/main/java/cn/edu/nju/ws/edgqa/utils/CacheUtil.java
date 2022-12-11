package cn.edu.nju.ws.edgqa.utils;

import cn.edu.nju.ws.edgqa.domain.beans.Link;

import java.io.ObjectOutputStream;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class CacheUtil {
    private static ObjectOutputStream dexterOutput;
    private static ObjectOutputStream earlOutput;
    private static ObjectOutputStream falconOutput;
    private static ObjectOutputStream relationSimilarityOutput;
    private static Map<String, Map<String, List<Link>>> dexterMap = new ConcurrentHashMap<>();
    private static Map<String, Map<String, List<Link>>> earlMap = new ConcurrentHashMap<>();
    private static Map<String, Map<String, List<Link>>> falconMap = new ConcurrentHashMap<>();

    public static ObjectOutputStream getDexterOutput() {
        return dexterOutput;
    }

    public static void setDexterOutput(ObjectOutputStream dexterOutput) {
        CacheUtil.dexterOutput = dexterOutput;
    }

    public static ObjectOutputStream getEarlOutput() {
        return earlOutput;
    }

    public static void setEarlOutput(ObjectOutputStream earlOutput) {
        CacheUtil.earlOutput = earlOutput;
    }

    public static ObjectOutputStream getFalconOutput() {
        return falconOutput;
    }

    public static void setFalconOutput(ObjectOutputStream falconOutput) {
        CacheUtil.falconOutput = falconOutput;
    }

    public static ObjectOutputStream getRelationSimilarityOutput() {
        return relationSimilarityOutput;
    }

    public static void setRelationSimilarityOutput(ObjectOutputStream relationSimilarityOutput) {
        CacheUtil.relationSimilarityOutput = relationSimilarityOutput;
    }

    public static Map<String, Map<String, List<Link>>> getDexterMap() {
        return dexterMap;
    }

    public static void setDexterMap(Map<String, Map<String, List<Link>>> dexterMap) {
        CacheUtil.dexterMap = dexterMap;
    }

    public static Map<String, Map<String, List<Link>>> getEarlMap() {
        return earlMap;
    }

    public static void setEarlMap(Map<String, Map<String, List<Link>>> earlMap) {
        CacheUtil.earlMap = earlMap;
    }

    public static Map<String, Map<String, List<Link>>> getFalconMap() {
        return falconMap;
    }

    public static void setFalconMap(Map<String, Map<String, List<Link>>> falconMap) {
        CacheUtil.falconMap = falconMap;
    }
}
