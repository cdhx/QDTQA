package cn.edu.nju.ws.edgqa.main;

import cn.edu.nju.ws.edgqa.domain.beans.Link;
import cn.edu.nju.ws.edgqa.domain.beans.LinkMap;
import cn.edu.nju.ws.edgqa.domain.edg.SparqlGenerator;
import cn.edu.nju.ws.edgqa.eval.CumulativeIRMetrics;
import cn.edu.nju.ws.edgqa.eval.Evaluator;
import cn.edu.nju.ws.edgqa.eval.IRMetrics;
import cn.edu.nju.ws.edgqa.handler.QASystem;
import cn.edu.nju.ws.edgqa.utils.FileUtil;
import cn.edu.nju.ws.edgqa.utils.LogUtil;
import cn.edu.nju.ws.edgqa.utils.UriUtil;
import cn.edu.nju.ws.edgqa.utils.enumerates.*;
import cn.edu.nju.ws.edgqa.utils.kbutil.KBUtil;
import cn.edu.nju.ws.edgqa.utils.linking.LinkingTool;
import org.apache.commons.cli.Options;
import org.apache.jena.query.Query;
import org.apache.jena.query.QueryFactory;
import org.apache.jena.query.Syntax;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import org.json.JSONArray;
import org.json.JSONObject;

import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.*;

public class PointerNetworkQA extends QASystem {
    private static JSONArray lcQuadSparqlArray = new JSONArray(FileUtil.readFileAsString("src/main/resources/datasets/lcquad-test.json"));
    private static JSONArray QALD9SparqlArray = new JSONArray(FileUtil.readFileAsString("src/main/resources/datasets/qald-9-test-en.json"));

    public static void main(String[] args) throws IOException {
        Options options = QAArgs.getOptions();
        int run = QAArgs.setArguments(options, args);

        if (run == 0) {
            runAutoTest();
        }
    }

    private static String getLogFileName() {
        Date date = new Date();
        SimpleDateFormat dataFormat = new SimpleDateFormat("yyyy_MM_dd HH_mm");
        String timeString = dataFormat.format(date);
        String logFileName = "pointer_network_" + QAArgs.getDatasetName() + "_" + timeString;
        if (QAArgs.getGoldenLinkingMode() == GoldenMode.FILTERING) {
            logFileName += "_golden_filtering";
        } else if (QAArgs.getGoldenLinkingMode() == GoldenMode.GENERATION) {
            logFileName += "_golden_generation";
        }
        return logFileName;
    }

    private static JSONArray getDataArray() {
        JSONArray dataArray = null;
        if (QAArgs.getDataset() == DatasetEnum.LC_QUAD) {
            dataArray = new JSONArray(FileUtil.readFileAsString("src/main/resources/datasets/pointer-network-lcquad-test.json"));
        } else if (QAArgs.getDataset() == DatasetEnum.QALD_9) {
            dataArray = new JSONArray(FileUtil.readFileAsString("src/main/resources/datasets/pointer-network-qald-9-test.json"));
        }
        return dataArray;
    }

    @Nullable
    private static String getGoldenSparql(DatasetEnum dataset, int serialNum) {
        if (dataset == DatasetEnum.LC_QUAD) {
            for (int i = 0; i < lcQuadSparqlArray.length(); i++) {
                JSONObject quesJSONObject = lcQuadSparqlArray.getJSONObject(i);
                if (quesJSONObject.getInt("_id") == serialNum) {
                    return quesJSONObject.getString("sparql_query");
                }
            }
        } else if (dataset == DatasetEnum.QALD_9) {
            for (int i = 0; i < QALD9SparqlArray.length(); i++) {
                JSONObject quesJSONObject = QALD9SparqlArray.getJSONObject(i);
                if (quesJSONObject.getInt("id") == serialNum) {
                    return quesJSONObject.getString("sparql_query");
                }
            }
        }
        return null;
    }

    public static void runAutoTest() throws IOException {
        long startTime = System.currentTimeMillis(); // for timing

        String logFileName = getLogFileName();  // get log file name
        LogUtil.addWriter("queryLog", "baseline_query_logs/query_log_" + logFileName + ".txt", true);
        LogUtil.addWriter("ansErrLog", "baseline_query_logs/ERR_Question_" + logFileName + ".txt", false);

        CumulativeIRMetrics cumulativeIRMetrics = new CumulativeIRMetrics();   // precision, recall, micro F1, macro F1
        CumulativeIRMetrics cumulativeQALDMetrics = new CumulativeIRMetrics(); // for QALD measure
        JSONArray dataArray;
        dataArray = getDataArray();
        LogUtil.printlnInfo("queryLog", "System initialization time: " + (System.currentTimeMillis() - startTime) + " ms");
        assert dataArray != null;

        // the QA process is here
        answerQuestions(startTime, cumulativeIRMetrics, cumulativeQALDMetrics, dataArray, 0, dataArray.length());

        LogUtil.printlnInfo("queryLog", "QuestionSolver total time: " + (System.currentTimeMillis() - startTime) + " ms");

        postProcess();
        LogUtil.printlnInfo("queryLog", "query generation finished");
        LogUtil.closeAllWriters();  // close the file writers
    }

    private static void answerQuestions(long startTime, CumulativeIRMetrics cumulativeIRMetrics, CumulativeIRMetrics cumulativeQALDMetrics, JSONArray dataArray, int quesIdBegin, int quesIdEnd) {
        for (int quesIdx = quesIdBegin; quesIdx < quesIdEnd; ++quesIdx) { // for each question
            try {
                long questionStartTime = System.currentTimeMillis();
                JSONObject quesJSONObj = dataArray.getJSONObject(quesIdx);

                String question = quesJSONObj.getString("question");
                String part1 = quesJSONObj.getString("split_part1");
                String part2 = quesJSONObj.getString("split_part2");
                String serialNumberStr = quesJSONObj.getString("ID");
                if (serialNumberStr.startsWith("LC-TEST")) {
                    serialNumberStr = serialNumberStr.substring(7);
                } else if (serialNumberStr.startsWith("QALD-Test")) {
                    serialNumberStr = serialNumberStr.substring(9);
                }
                String goldenSparql = getGoldenSparql(QAArgs.getDataset(), Integer.parseInt(serialNumberStr));

                LogUtil.println("queryLog", "\n\nQuestion serial " + serialNumberStr + ": " + question, LogType.None);
                // generate golden answer from golden sparql
                Query goldenQuery = QueryFactory.create(goldenSparql, Syntax.syntaxARQ);
                List<String> goldenAnswer = new ArrayList<>(KBUtil.getQueryStringResult(goldenQuery));  // the golden answers
                List<String> predictedAnswer = solver(question, part1, part2, getComplexQuestionType(quesJSONObj.getString("comp")));

                LogUtil.printlnInfo("queryLog", "Golden sparql: " + goldenSparql);

                LogUtil.printlnInfo("queryLog", "Golden Answer: " + goldenAnswer);
                LogUtil.printlnInfo("queryLog", "Predicted Answer: " + predictedAnswer);

                // precision, recall and F1 for this question
                IRMetrics localIRMetrics = Evaluator.getMetrics(predictedAnswer, goldenAnswer);
                cumulativeIRMetrics.addSample(localIRMetrics);
                LogUtil.printlnInfo("queryLog", localIRMetrics.toString());
                LogUtil.printlnInfo("queryLog", cumulativeIRMetrics.toString());

                if (cumulativeQALDMetrics != null) {
                    IRMetrics localQALDIRMetrics = Evaluator.getQALDMetrics(predictedAnswer, goldenAnswer);
                    cumulativeQALDMetrics.addSample(localQALDIRMetrics);
                    LogUtil.printlnInfo("queryLog", cumulativeQALDMetrics.toQALDString());
                }

            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    /**
     * Get the ComplexQuestionType
     *
     * @param comp the string from json file
     * @return an instance of ComplexQuestionType
     */
    public static ComplexQuestionType getComplexQuestionType(String comp) {
        if (comp.equals("conjunction"))
            return ComplexQuestionType.CONJUNCTION;
        if (comp.equals("composition"))
            return ComplexQuestionType.COMPOSITION;
        return ComplexQuestionType.UNKNOWN;
    }

    /**
     * Question solver for PointerNetwork splitting model
     *
     * @param question            the whole natural language question
     * @param part1               question part 1
     * @param part2               question part 2
     * @param complexQuestionType the complex type
     * @return the answer list
     */
    @NotNull
    public static List<String> solver(String question, String part1, String part2, ComplexQuestionType complexQuestionType) {
        List<String> part1Ans = solver(part1);
        if (part1Ans.isEmpty()) { // no answer for part1, try answering the whole question directly
            return solver(question);
        }
        List<String> finalAns = new ArrayList<>();
        if (complexQuestionType == ComplexQuestionType.CONJUNCTION) {
            List<String> part2Ans = solver(part2);
            finalAns = new ArrayList<>(part1Ans);
            finalAns.retainAll(part2Ans);
        } else if (complexQuestionType == ComplexQuestionType.COMPOSITION) {
            while (part1Ans.size() > 20) {
                part1Ans.remove(20);
            }
            Set<String> part2Questions = getCompositionQuestions(part2, part1Ans);
            for (String ques : part2Questions) {
                finalAns.addAll(solver(ques));
            }
        }
        return finalAns;
    }

    public static Set<String> getCompositionQuestions(String part2, List<String> part1Ans) {
        Set<String> res = new HashSet<>();
        if (!part2.contains("%composition")) {
            res.add(part2);
            return res;
        }
        for (String ans : part1Ans) {
            String newQues = part2;
            newQues = newQues.replace("%composition", UriUtil.extractUri(ans));
            res.add(newQues);
        }
        return res;
    }

    public static List<String> solver(String question) {
        List<String> ans = new ArrayList<>();
        question = question.trim();
        if (question.isEmpty())
            return ans;

        try {
            LinkMap entityLinkMap = new LinkMap();
            LinkMap relationLinkMap = new LinkMap();
            LinkingTool.getEnsembleLinking(question, entityLinkMap.getData(), relationLinkMap.getData(), true);
            Link entityLink = entityLinkMap.topLink();
            if (entityLink == null)
                return ans;
            Link relationLink = relationLinkMap.oneHopTopLink(entityLink.getUri());
            if (relationLink == null)
                return ans;

            SparqlGenerator sparqlGen = new SparqlGenerator(getQueryType(question));
            sparqlGen.setFinalVarName("uri");
            sparqlGen.addTriple("?uri", relationLink.getUri(), entityLink.getUri());

            List<SparqlGenerator> sparqlGeneratorList = sparqlGen.expandQueryWithDbpOrDbo();

            for (SparqlGenerator sparqlGenerator : sparqlGeneratorList) {
                ans.addAll(sparqlGenerator.solve());
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return ans;
    }

    public static QueryType getQueryType(String question) {
        if (question.isEmpty())
            return QueryType.UNKNOWN;
        question = question.trim().toLowerCase();
        if (question.contains("how many") || question.startsWith("count") || question.contains("total number of")
                || question.contains("what is the number of"))
            return QueryType.COUNT;
        if (question.startsWith("was") || question.startsWith("does") || question.startsWith("is") || question.startsWith("did"))
            return QueryType.JUDGE;
        return QueryType.COMMON;
    }

}
