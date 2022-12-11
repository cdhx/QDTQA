package cn.edu.nju.ws.edgqa.eval;

import cn.edu.nju.ws.edgqa.domain.beans.Link;
import cn.edu.nju.ws.edgqa.utils.CollectionUtil;
import cn.edu.nju.ws.edgqa.utils.enumerates.QueryType;

import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

public class Evaluator {

    /**
     * IR metrics for entity linking
     */
    private static CumulativeIRMetrics entityLinkingMetrics = new CumulativeIRMetrics();

    /**
     * IR metrics for relation linking
     */
    private static CumulativeIRMetrics relationLinkingMetrics = new CumulativeIRMetrics();

    /**
     * IR metrics for type linking
     */
    private static CumulativeIRMetrics typeLinkingMetrics = new CumulativeIRMetrics();

    /**
     * Hit rate for query type identification
     */
    private static Map<QueryType, CumulativeHitRate> queryTypeHitRate = new HashMap<>();

    /**
     * Hit rate for question hop identification
     * The number of hops should be the number of blocks
     */
    private static Map<Integer, CumulativeHitRate> hopHitRate = new HashMap<>();

    /**
     * Hit rate for the number of triples
     * The number of triples should be the number of descriptive node
     */
    private static Map<Integer, CumulativeHitRate> tripleNumHitRate = new HashMap<>();

    /**
     * IR metrics for query type
     */
    private static Map<QueryType, CumulativeIRMetrics> queryTypeMetricsMap = new HashMap<>();

    /**
     * IR metrics for SPARQL template
     */
    private static Map<Integer, CumulativeIRMetrics> templateMetricsMap = new HashMap<>();

    /**
     * IR metrics for the number of triples
     */
    private static Map<Integer, CumulativeIRMetrics> tripleNumMetricsMap = new HashMap<>();

    /**
     * IR metrics for the number of hops
     */
    private static Map<Integer, CumulativeIRMetrics> hopMetricsMap = new HashMap<>();

    public static void addQueryTypeHit(QueryType golden, QueryType prediction) {
        queryTypeHitRate.computeIfAbsent(golden, k -> new CumulativeHitRate());
        boolean equal;
        if (golden == QueryType.COMMON && prediction != QueryType.JUDGE && prediction != QueryType.COUNT)
            equal = true;
        else
            equal = (golden == prediction);
        queryTypeHitRate.get(golden).addSample(equal);
    }

    public static String getQueryTypeHitStr() {
        StringBuilder stringBuilder = new StringBuilder();
        for (QueryType key : queryTypeHitRate.keySet()) {
            stringBuilder.append("query type: ").append(key).append(", sample: ").append(queryTypeHitRate.get(key).getNumGoldenSample()).append(", hit rate: ").append(queryTypeHitRate.get(key).getHitRate()).append("\n");
        }
        stringBuilder.deleteCharAt(stringBuilder.length() - 1); // remove "\n"
        return stringBuilder.toString();
    }

    /**
     * Add a sample of hop identification
     *
     * @param golden     the number of hops in golden sparql
     * @param prediction the number of EDG blocks
     */
    public static void addHopHit(int golden, int prediction) {
        hopHitRate.computeIfAbsent(golden, k -> new CumulativeHitRate());
        hopHitRate.get(golden).addSample((golden == prediction));
    }

    public static String getHopHitStr() {
        StringBuilder stringBuilder = new StringBuilder();
        for (Integer key : hopHitRate.keySet()) {
            stringBuilder.append("hop: ").append(key).append(", sample: ").append(hopHitRate.get(key).getNumGoldenSample()).append(", hit rate: ").append(hopHitRate.get(key).getHitRate()).append("\n");
        }
        stringBuilder.deleteCharAt(stringBuilder.length() - 1); // remove "\n"
        return stringBuilder.toString();
    }

    public static void addTripleNumHit(int golden, int prediction) {
        tripleNumHitRate.computeIfAbsent(golden, k -> new CumulativeHitRate());
        tripleNumHitRate.get(golden).addSample((golden == prediction));
    }

    public static String getTripleNumHitStr() {
        StringBuilder stringBuilder = new StringBuilder();
        for (Integer key : tripleNumHitRate.keySet()) {
            stringBuilder.append("triple number: ").append(key).append(", sample: ").append(tripleNumHitRate.get(key).getNumGoldenSample()).append(", hit rate: ").append(tripleNumHitRate.get(key).getHitRate()).append("\n");
        }
        stringBuilder.deleteCharAt(stringBuilder.length() - 1); // remove "\n"
        return stringBuilder.toString();
    }

    public static void addEntityLinkingSample(List<String> prediction, List<String> golden) {
        entityLinkingMetrics.addSample(getMetrics(prediction, golden));
    }

    public static void addRelationLinkingSample(List<String> prediction, List<String> golden) {
        relationLinkingMetrics.addSample(getMetrics(prediction, golden));
    }

    public static void addTypeLinkingSample(List<String> prediction, List<String> golden) {
        typeLinkingMetrics.addSample(getMetrics(prediction, golden));
    }

    public static String getEntityLinkingMetricsStr() {
        try {
            return "Entity Linking P: " + entityLinkingMetrics.getPrecision() + ", R: " + entityLinkingMetrics.getRecall() +
                    ", micro F1: " + entityLinkingMetrics.getAverageF1() + ", macro F1: "
                    + entityLinkingMetrics.getMacroF1();
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    public static String  getRelationLinkingMetricsStr() {
        try {
            return "Relation Linking P: " + relationLinkingMetrics.getPrecision() + ", R: " + relationLinkingMetrics.getRecall() +
                    ", micro F1: " + relationLinkingMetrics.getAverageF1() + ", macro F1: "
                    + relationLinkingMetrics.getMacroF1();
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    public static String getTypeLinkingMetricsStr() {
        try {
            return "Type Linking P: " + typeLinkingMetrics.getPrecision() + ", R: " + typeLinkingMetrics.getRecall() +
                    ", micro F1: " + typeLinkingMetrics.getAverageF1() + ", macro F1: "
                    + typeLinkingMetrics.getMacroF1();
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    /**
     * Get the metrics, precision, recall and F1
     *
     * @param prediction prediction list
     * @param golden     golden answer list
     * @return the metrics, including precision, recall and micro F1
     */
    public static IRMetrics getMetrics(List<String> prediction, List<String> golden) {
        IRMetrics res = new IRMetrics();
        // null check
        if (prediction == null || golden == null)
            return res;

        // convert to labels
        Set<String> predictionSet = answerListToSet(prediction);
        Set<String> goldenSet = answerListToSet(golden);

        if (goldenSet.isEmpty() && predictionSet.isEmpty()) {
            res.setMetrics(1.0, 1.0, 1.0);
            return res;
        }

        if (goldenSet.isEmpty() && !predictionSet.isEmpty()) {
            res.setMetrics(0, 0, 0);
            return res;
        }

        if (!golden.isEmpty() && predictionSet.isEmpty()) {
            res.setMetrics(0, 0, 0);
            return res;
        }


        if (predictionSet.size() == goldenSet.size() && predictionSet.equals(goldenSet)) {
            res.setMetrics(1.0, 1.0, 1.0);
            return res;
        }

        int hit = CollectionUtil.setIntersect(predictionSet, goldenSet).size();

        double precision = (double) hit / (double) predictionSet.size();
        double recall = (double) hit / (double) goldenSet.size();

        res.setMetrics(precision, recall);
        return res;
    }

    /**
     * return the QALD metrics
     *
     * @param prediction prediction answers
     * @param golden     golden answers
     * @return QALD metric
     */
    public static IRMetrics getQALDMetrics(List<String> prediction, List<String> golden) {
        if (golden.isEmpty() && prediction.isEmpty()) {
            return new IRMetrics(1, 1, 1);
        } else if (!golden.isEmpty() && prediction.isEmpty()) {
            return new IRMetrics(1, 0, 0);
        } else {
            return getMetrics(prediction, golden);
        }
    }

    /**
     * Convert the answers to standard form, return an answer set
     *
     * @param answerList answer List
     * @return standardized answer set
     */
    public static Set<String> answerListToSet(List<String> answerList) {

        Set<String> res = new HashSet<>();
        answerList.forEach(answer -> {
            String[] splitArr = answer.split("/");
            String label = splitArr[splitArr.length - 1].replace("_", " ").replace("@en", "");
            if (label.length() == 10 && label.endsWith("-01-01")) {
                label = label.substring(0, 4);
            }

            Matcher matcher = Pattern.compile("(\\d+)-(\\d+)-(\\d+)").matcher(label);
            if (matcher.matches()) {
                //System.out.println(label);
                String year = matcher.group(1);
                String month = matcher.group(2);
                String day = matcher.group(3);
                while (year.length() < 4) {
                    year = "0" + year;
                }
                while (month.length() < 2) {
                    month = "0" + month;
                }
                while (day.length() < 2) {
                    day = "0" + day;
                }
                label = year + "-" + month + "-" + day;
            }

            res.add(label);
        });
        return res;
    }

    /**
     * Get the metrics by list, precision, recall, micro F1
     *
     * @param prediction prediction link list
     * @param golden     golden link list
     * @return a list of double, [0] precision, [1] recall, [2] micro-F1
     */
    public static IRMetrics getMetricsByLinkList(List<Link> prediction, List<Link> golden) {
        List<String> predictionStrList = new ArrayList<>();
        List<String> goldenStrList = new ArrayList<>();

        if (prediction != null) {
            for (Link link : prediction) {
                predictionStrList.add(link.getUri());
            }
        }
        if (golden != null) {
            for (Link link : golden) {
                goldenStrList.add(link.getUri());
            }
        }
        predictionStrList = predictionStrList.stream().distinct().collect(Collectors.toList());
        goldenStrList = goldenStrList.stream().distinct().collect(Collectors.toList());

        return getMetrics(predictionStrList, goldenStrList);
    }

    public static IRMetrics getMetricsByLinkMap(Map<String, List<Link>> prediction, List<Link> golden) {
        List<Link> linkList = new ArrayList<>();
        for (Map.Entry<String, List<Link>> entry : prediction.entrySet()) {
            linkList.addAll(entry.getValue());
        }
        linkList = linkList.stream().distinct().collect(Collectors.toList());
        return getMetricsByLinkList(linkList, golden);
    }

    public static void addSampleByQuesType(QueryType quesType, IRMetrics metrics) {
        if (!queryTypeMetricsMap.containsKey(quesType)) {
            queryTypeMetricsMap.put(quesType, new CumulativeIRMetrics());
        }
        queryTypeMetricsMap.get(quesType).addSample(metrics);
    }

    public static void addSampleByTemplate(Integer templateId, IRMetrics metrics) {
        if (!templateMetricsMap.containsKey(templateId)) {
            templateMetricsMap.put(templateId, new CumulativeIRMetrics());
        }
        templateMetricsMap.get(templateId).addSample(metrics);
    }

    /**
     * Add a sample by the number of triples
     *
     * @param tripleNum the number of triples
     * @param metrics   IR metrics for a question
     */
    public static void addSampleByTripleNum(Integer tripleNum, IRMetrics metrics) {
        if (!tripleNumMetricsMap.containsKey(tripleNum)) {
            tripleNumMetricsMap.put(tripleNum, new CumulativeIRMetrics());
        }
        tripleNumMetricsMap.get(tripleNum).addSample(metrics);
    }

    /**
     * Add a sample by the number of hops
     *
     * @param hop     the number of hops
     * @param metrics IR metrics for a question
     */
    public static void addSampleByHop(Integer hop, IRMetrics metrics) {
        if (!hopMetricsMap.containsKey(hop)) {
            hopMetricsMap.put(hop, new CumulativeIRMetrics());
        }
        hopMetricsMap.get(hop).addSample(metrics);
    }

    public static String getQuesTypeMetricsStr() {
        StringBuilder stringBuilder = new StringBuilder();
        for (QueryType queryType : queryTypeMetricsMap.keySet()) {
            stringBuilder.append("question type metrics ").append(queryType).append(", ").append(queryTypeMetricsMap.get(queryType).toString()).append("\n");
        }
        stringBuilder.deleteCharAt(stringBuilder.length() - 1); // remove "\n"
        return stringBuilder.toString();
    }

    public static String getTemplateMetricsStr() {
        StringBuilder stringBuilder = new StringBuilder();
        for (Integer templateId : templateMetricsMap.keySet()) {
            stringBuilder.append("template metrics ").append(templateId).append(", ").append(templateMetricsMap.get(templateId).toString()).append("\n");
        }
        stringBuilder.deleteCharAt(stringBuilder.length() - 1); // remove "\n"
        return stringBuilder.toString();
    }

    public static String getTripleNumMetricsStr() {
        StringBuilder stringBuilder = new StringBuilder();
        for (Integer tripleNum : tripleNumMetricsMap.keySet()) {
            stringBuilder.append("triple number metrics ").append(tripleNum).append(", ").append(tripleNumMetricsMap.get(tripleNum).toString()).append("\n");
        }
        stringBuilder.deleteCharAt(stringBuilder.length() - 1); // remove "\n"
        return stringBuilder.toString();
    }

    public static String getHopMetricsStr() {
        StringBuilder stringBuilder = new StringBuilder();
        for (Integer hop : hopMetricsMap.keySet()) {
            stringBuilder.append("hop metrics ").append(hop).append(", ").append(hopMetricsMap.get(hop).toString()).append("\n");
        }
        stringBuilder.deleteCharAt(stringBuilder.length() - 1); // remove "\n"
        return stringBuilder.toString();
    }
}
