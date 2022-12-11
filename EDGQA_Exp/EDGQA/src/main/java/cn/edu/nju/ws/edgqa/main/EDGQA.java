package cn.edu.nju.ws.edgqa.main;

import cn.edu.nju.ws.edgqa.domain.beans.tuple.TwoTuple;
import cn.edu.nju.ws.edgqa.domain.edg.EDG;
import cn.edu.nju.ws.edgqa.domain.edg.Preprocessor;
import cn.edu.nju.ws.edgqa.domain.edg.SparqlGenerator;
import cn.edu.nju.ws.edgqa.eval.CumulativeIRMetrics;
import cn.edu.nju.ws.edgqa.eval.Evaluator;
import cn.edu.nju.ws.edgqa.eval.IRMetrics;
import cn.edu.nju.ws.edgqa.handler.Detector;
import cn.edu.nju.ws.edgqa.handler.QASystem;
import cn.edu.nju.ws.edgqa.handler.QuestionSolver;
import cn.edu.nju.ws.edgqa.utils.FileUtil;
import cn.edu.nju.ws.edgqa.utils.LogUtil;

import cn.edu.nju.ws.edgqa.utils.Timer;
import cn.edu.nju.ws.edgqa.utils.enumerates.DatasetEnum;
import cn.edu.nju.ws.edgqa.utils.enumerates.GoldenMode;
import cn.edu.nju.ws.edgqa.utils.enumerates.LogType;
import cn.edu.nju.ws.edgqa.utils.enumerates.QueryType;
import cn.edu.nju.ws.edgqa.utils.kbutil.KBUtil;
import org.apache.commons.cli.Options;
import org.apache.commons.io.FileUtils;
import org.apache.jena.query.Query;
import org.apache.jena.query.QueryFactory;
import org.apache.jena.query.Syntax;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintStream;
import java.io.File;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.concurrent.ExecutorService;

/**
 * The main class for the QA system
 */
public class EDGQA extends QASystem {


    /**
     * The question solver main function
     *
     * @param args the arguments, it is processed by Apache Commons Command-CLI
     * @throws IOException the IO exception
     */
    public static void main(String[] args) throws IOException {
        Options options = QAArgs.getOptions();
        int run = QAArgs.setArguments(options, args);

        switch (run) {
            case 0:
                runAutoTest();
                break;
            case 1:
                runQA();
                break;
            case 2:
                runTestBySerialNumber();
                break;
            case 3:
                runTestByQuestionId();
                break;
            case 4:
                runAutoCandTest();
                break;
        }
    }

    /**
     * Answer testset questions by EDG
     *
     * @throws IOException the IO exception
     */
    public static void runAutoTest() throws IOException {
        long startTime = System.currentTimeMillis(); // for timing
        String isTrain = QAArgs.isIsTraining() ? "train" : "test";

        String logFileName = getLogFileName();  // get log file name
        LogUtil.addWriter("queryLog", "query_logs/query_log_" + isTrain + "_" + logFileName + ".txt", true);

        CumulativeIRMetrics cumulativeIRMetrics = new CumulativeIRMetrics(); // precision, recall, micro F1, macro F1
        CumulativeIRMetrics qaldCumulativeIRMetrics = new CumulativeIRMetrics(); // QALD precision, recall, micro F1, QALD macro F1
        JSONArray dataArray;
        dataArray = getDataArray();
        LogUtil.printlnInfo("queryLog", "System initialization time: " + (System.currentTimeMillis() - startTime) + " ms");
        assert dataArray != null;
        List<String> questionList = null;

        // the QA process is here
        answerQuestions(startTime, cumulativeIRMetrics, qaldCumulativeIRMetrics, dataArray, 0, dataArray.length(), questionList);

        LogUtil.printlnInfo("queryLog", "QuestionSolver total time: " + (System.currentTimeMillis() - startTime) + " ms");

        QASystem.postProcess();
        LogUtil.printlnInfo("queryLog", "query generation finished");
        LogUtil.closeAllWriters();  // close the file writers
    }

    /**
     * Answer testset questions by EDG, evaluate best F1
     *
     * @throws IOException the IO exception
     */
    public static void runAutoCandTest() throws IOException {

        // candidate test, do not reduce candidates
        QAArgs.setEvaluateCandNum(true);
        // for timing
        long startTime = System.currentTimeMillis();

        // set large candidate limit
        QAArgs.setCandidateSparqlNumUpperB(30);
        QAArgs.setRootCandidateSparqlNumUpperB(30);

        JSONArray dataArray;
        dataArray = getDataArray();
        assert dataArray != null;
        // the QA process is here
        evaluateCandidate(dataArray, 0, dataArray.length());

        System.out.println("Time Consumed:" + (System.currentTimeMillis() - startTime));

        QASystem.postProcess();
    }

    /**
     * get the json array of QA dataset
     *
     * @return json array of QA dataset
     */
    @Nullable
    private static JSONArray getDataArray() {
        JSONArray sparqlArray = null;
        if (QAArgs.getDataset() == DatasetEnum.LC_QUAD) {
            //
            if (QAArgs.isIsTraining()) {
                sparqlArray = new JSONArray(FileUtil.readFileAsString("src/main/resources/datasets/lcquad-train.json"));
            } else {
                sparqlArray = new JSONArray(FileUtil.readFileAsString("src/main/resources/datasets/lcquad-test.json"));
            }
        } else if (QAArgs.getDataset() == DatasetEnum.QALD_9) {
            if (QAArgs.isIsTraining()) {
                sparqlArray = new JSONArray(FileUtil.readFileAsString("src/main/resources/datasets//qald-9-train-en.json"));
            } else {
                sparqlArray = new JSONArray(FileUtil.readFileAsString("src/main/resources/datasets//qald-9-test-en.json"));
            }
        }
        return sparqlArray;
    }

    @NotNull
    public static String getLogFileName() {
        Date date = new Date();
        SimpleDateFormat dataFormat = new SimpleDateFormat("yyyy_MM_dd HH_mm");
        String timeString = dataFormat.format(date);
        String logFileName = QAArgs.fileToRun.substring(QAArgs.fileToRun.lastIndexOf("/")+1) + "_" +QAArgs.getDatasetName() + "_" + timeString;
        if (QAArgs.getGoldenLinkingMode() == GoldenMode.FILTERING) {
            logFileName += "_golden_filtering";
        } else if (QAArgs.getGoldenLinkingMode() == GoldenMode.GENERATION) {
            logFileName += "_golden_generation";
        }
        return logFileName;
    }

    private static String questionProcess(String str) {
        if (str == null || str.isEmpty())
            return str;
        str = str.trim().toLowerCase();
        if (str.endsWith("?") || str.endsWith(".")) {
            str = str.substring(0, str.length() - 1).trim();
        }
        return str;
    }

    public static ExecutorService getPool() {
        return QASystem.pool;
    }

    /**
     * answer a set of questions, it's an import loop in the autotest
     *
     * @param startTime           system start time
     * @param cumulativeIRMetrics evaluation metrics
     * @param sparqlArray         the array of sparql
     * @param quesIdBegin         the beginning of the question id interval [begin, end)
     * @param quesIdEnd           the end of the question id interval [begin, end)
     * @param questionList        the list of questions to be answered, default: null
     * @throws IOException the IO exception
     */
    private static void answerQuestions(long startTime, CumulativeIRMetrics cumulativeIRMetrics, CumulativeIRMetrics QALDcumulativeIRMetrics, JSONArray sparqlArray, int quesIdBegin, int quesIdEnd, List<String> questionList) throws IOException {
        SimpleDateFormat sdf = new SimpleDateFormat("MM_dd_HH_mm");
        String isTrain = QAArgs.isIsTraining() ? "train" : "test";
        if (questionList != null && !questionList.isEmpty()) {
            LogUtil.printlnInfo("queryLog", "question list len: " + questionList.size());
        }
        //JSONArray edgJsonArray = new JSONArray();
        File file = new File(QAArgs.fileToRun);
        String content = FileUtils.readFileToString(file);
        // load edgs from json
        JSONArray edgsFromJSON = new JSONArray(content);
        boolean isGeneratingSubScore = false;
        for (int quesIdx = quesIdBegin; quesIdx < quesIdEnd; ++quesIdx) { // for each question
            long questionStartTime = System.currentTimeMillis();
            //edg of the question
            JSONObject quesJSONObj = sparqlArray.getJSONObject(quesIdx);

            int serialNumber = -1, sparqlTemplateId = -1;
            String question = null;
            String goldenSparql = null;
            List<String> goldenAnswer = new ArrayList<>();

            if (QAArgs.getDataset() == DatasetEnum.LC_QUAD) {
                serialNumber = quesJSONObj.getInt("_id");
                sparqlTemplateId = quesJSONObj.getInt("sparql_template_id");

                // the original question
                question = quesJSONObj.getString("corrected_question");
                // golden sparql and rewrite
                goldenSparql = quesJSONObj.getString("sparql_query");
                goldenSparql = goldenSparql.replaceAll(" COUNT\\(", " COUNT(DISTINCT ");
                // generate gold answer from golden sparql
                Query goldenQuery = QueryFactory.create(goldenSparql, Syntax.syntaxARQ);
                // the golden answers
                goldenAnswer.addAll(KBUtil.getQueryStringResult(goldenQuery));

            } else if (QAArgs.getDataset() == DatasetEnum.QALD_9) {

                serialNumber = quesJSONObj.getInt("id");
                question = quesJSONObj.getString("question");
                goldenSparql = quesJSONObj.getString("sparql_query");

                try {
                    goldenSparql = Preprocessor.rewriteQALDSparql(goldenSparql);  // rewrite sparql for qald
                    Query goldenQuery = QueryFactory.create(goldenSparql, Syntax.syntaxARQ);
                    goldenAnswer.addAll(KBUtil.getQueryStringResult(goldenQuery)); // the golden answers
                } catch (Exception e) {
                    System.out.println("[WARN] " + QAArgs.getDatasetName() + " question: " + quesIdx + " sparql exception, the golden answer will be used");
                    quesJSONObj.getJSONArray("answers").toList().forEach(o -> goldenAnswer.add(o.toString()));
                }
            }

            assert question != null;
            // filter by question list
            if (questionList != null && !questionList.isEmpty()) {
                if (!questionList.contains(questionProcess(question))) {  // not in the list, continue
                    System.out.println("question pass " + question);
                    continue;
                }
            }

            EDG edg = null;
            QuestionSolver questionSolver = null;
            List<String> predictedAnswer = new ArrayList<>(); // the generated answers by EDG

            // generate answers based on EDG
            try {
                LogUtil.println("queryLog", "\n\nQuestion " + quesIdx + ": " + question, LogType.None);

                long startEDGGenTime = System.currentTimeMillis();
                //edg = new EDG(question); // generate the EDG
                edg = edg.fromJSON(edgsFromJSON.getJSONObject(quesIdx-quesIdBegin));
               //JSONObject edgJ = edg.toJSON();
               //edgJ.put("id", quesIdx);
               //edgJsonArray.put(edgJ);
                Timer.totalEDGGenTime += (System.currentTimeMillis() - startEDGGenTime);

                questionSolver = new QuestionSolver(question, serialNumber, edg);
                predictedAnswer.addAll(questionSolver.solveQuestion()); // generate our answers here

                LogUtil.printlnDebug("queryLog", "Golden sparql: " + goldenSparql);
                LogUtil.printlnInfo("queryLog", "SparqlGenerator list: " + questionSolver.getSubQuerySparqlMap().get(0));

                Map<Integer, List<SparqlGenerator>> subQuerySparqlMap = questionSolver.getSubQuerySparqlMap();

                if (isGeneratingSubScore) {
                    Map<Integer, List<TwoTuple<SparqlGenerator, Double>>> subQueryScoreMap = new HashMap<>();

                    int maxentityID = Collections.max(subQuerySparqlMap.keySet());
                    for (int entityID = 0; entityID <= maxentityID; entityID++) {

                        List<SparqlGenerator> sparqlGeneratorList = subQuerySparqlMap.get(entityID);

                        List<TwoTuple<SparqlGenerator, Double>> subQueryScoreList = new ArrayList<>();
                        for (SparqlGenerator spg : sparqlGeneratorList) {
                            if (entityID == 0) {
                                List<String> partialAnswer = KBUtil.getQueryStringResult(QueryFactory.create(spg.toSparql()));
                                IRMetrics metrics = Evaluator.getMetrics(partialAnswer, goldenAnswer);
                                double f1 = metrics.getMicroF1();
                                //scoreList.add(score);
                                subQueryScoreList.add(new TwoTuple<>(spg, f1));
                            } else {
                                List<TwoTuple<SparqlGenerator, Double>> twoTuples = subQueryScoreMap.get(0);
                                double max = 0;
                                for (TwoTuple<SparqlGenerator, Double> twoTuple : twoTuples) {
                                    if (twoTuple.getFirst().contains(spg)) {
                                        if (max < twoTuple.getSecond()) {
                                            max = twoTuple.getSecond();
                                        }
                                    }
                                }
                                subQueryScoreList.add(new TwoTuple<>(spg, max));
                            }
                        }
                        subQueryScoreMap.put(entityID, subQueryScoreList);

                    }

                    quesJSONObj.put("subqueries_score", subQueryScoreMap);
                }

                quesJSONObj.put("sub queries", subQuerySparqlMap);

                /*if (edg.getNumNode() <= 2) {
                    LogUtil.printlnInfo("edgErrLog", "Question " + quesIdx + ":" + question);
                }*/
            } catch (Exception e) {
                LogUtil.printlnError("queryLog", "Question " + quesIdx + " exception: answer generation");
                e.printStackTrace();
            }


            if (goldenAnswer.size() <= 1000) {
                LogUtil.printlnInfo("queryLog", "Golden Answer: " + goldenAnswer);
            } else {
                LogUtil.printlnInfo("queryLog", "Golden Answer: " + goldenAnswer.subList(0, 1000) + "...");
            }

            if (predictedAnswer.size() <= 1000) {
                LogUtil.printlnInfo("queryLog", "Predicted Answer: " + predictedAnswer);
            } else {
                LogUtil.printlnInfo("queryLog", "Predicted Answer: " + predictedAnswer.subList(0, 1000) + "...");
            }

            IRMetrics localIRMetrics = null;
            IRMetrics localQALDIRMetrics = null;
            if (QAArgs.getDataset() == DatasetEnum.QALD_9) {
                // precision, recall and F1 follow the setting of QALD
                localIRMetrics = Evaluator.getMetrics(predictedAnswer, goldenAnswer);
                localQALDIRMetrics = Evaluator.getQALDMetrics(predictedAnswer, goldenAnswer);
            } else {
                // precision, recall and F1 for this question
                localIRMetrics = Evaluator.getMetrics(predictedAnswer, goldenAnswer);
            }
            LogUtil.printlnInfo("queryLog", localIRMetrics.toString());

            computeMetrics(startTime, cumulativeIRMetrics, QALDcumulativeIRMetrics, quesIdx, questionStartTime, sparqlTemplateId, goldenSparql, edg, localIRMetrics, localQALDIRMetrics);

        }
        /*try{
            FileWriter  file = new FileWriter( "/home/home2/xxhu/EDGQA/EDG_lc_decom.json",false);
            BufferedWriter bfw = new BufferedWriter(new FileWriter("/home/home2/xxhu/EDGQA/EDG_lc_decom.json"));
            bfw.write(edgJsonArray.toString(4));
            bfw.flush();
            bfw.close();
        }
        catch(Exception e  ){

            e.getMessage();

        }
        */
    }

    private static void evaluateCandidate(JSONArray sparqlArray, int quesIDBegin, int quesIDEnd) throws IOException {

        JSONArray array = new JSONArray();
        BufferedWriter bfw1 = new BufferedWriter(new FileWriter("output/lctest_candidates.json"));

        for (int quesIdx = quesIDBegin; quesIdx < quesIDEnd; ++quesIdx) { // for each question
            System.out.println("Question :" + quesIdx);
            JSONObject quesJSONObj = sparqlArray.getJSONObject(quesIdx);

            int serialNumber = -1;
            String question = null;
            String intermediateQuestion = null;
            int template = -1;
            List<String> goldenAnswer = new ArrayList<>();
            String goldenSparql = quesJSONObj.getString("sparql_query");  // golden sparql
            goldenSparql = goldenSparql.replaceAll(" COUNT\\(", " COUNT(DISTINCT "); // count distinct
            System.out.println(goldenSparql);

            if (QAArgs.getDataset() == DatasetEnum.LC_QUAD) {
                serialNumber = quesJSONObj.getInt("_id");
                question = quesJSONObj.getString("corrected_question");  // the original question
                template = quesJSONObj.getInt("sparql_template_id");
                intermediateQuestion = quesJSONObj.getString("intermediary_question");


                // generate gold answer from golden sparql
                Query goldenQuery = QueryFactory.create(goldenSparql, Syntax.syntaxARQ);
                goldenAnswer.addAll(KBUtil.getQueryStringResult(goldenQuery)); // the golden answers

            } else if (QAArgs.getDataset() == DatasetEnum.QALD_9) {
                serialNumber = quesJSONObj.getInt("id");
                question = quesJSONObj.getString("question");

                try {
                    goldenSparql = Preprocessor.rewriteQALDSparql(goldenSparql);  // rewrite sparql for qald
                    Query goldenQuery = QueryFactory.create(goldenSparql, Syntax.syntaxARQ);
                    goldenAnswer.addAll(KBUtil.getQueryStringResult(goldenQuery)); // the golden answers
                } catch (Exception e) {
                    System.out.println("[WARN] " + QAArgs.getDatasetName() + " question: " + quesIdx + " sparql exception, the golden answer will be used");
                    quesJSONObj.getJSONArray("answers").toList().forEach(o -> goldenAnswer.add(o.toString()));
                }
            }

            assert question != null;
            EDG edg = null;
            QuestionSolver questionSolver = null;


            // generate answers based on EDG
            try {

                edg = new EDG(question); // generate the EDG
                questionSolver = new QuestionSolver(question, serialNumber, edg);
                questionSolver.solveQuestion();
                List<SparqlGenerator> sparqlList = questionSolver.getSubQuerySparqlMap().get(0);
                //System.out.println("Predicted Sparql:" + sparqlList);

                QueryType qType = edg.getQueryType();

                IRMetrics[] irMetrics = new IRMetrics[30];
                for (int i = 0; i < Math.min(sparqlList.size(), 30); i++) {
                    SparqlGenerator spg = sparqlList.get(i);
                    List<SparqlGenerator> tempList = spg.expandQueryWithDbpOrDbo();
                    //System.out.println("TempList:"+tempList);

                    List<String> ans = new ArrayList<>();
                    for (SparqlGenerator sparqlGenerator : tempList) {
                        ans.addAll(sparqlGenerator.solve());
                    }

                    if (qType == QueryType.JUDGE) {
                        if (ans.contains("true")) {
                            ans.clear();
                            ans.add("true");
                        } else {
                            ans.clear();
                            ans.add("false");
                        }
                    }

                    IRMetrics metrics = Evaluator.getMetrics(ans, goldenAnswer);
                    irMetrics[i] = metrics;
                }

                System.out.println("IRMetrics:" + Arrays.toString(irMetrics));

                if (qType != QueryType.JUDGE) {// exclude judge
                    JSONObject edgObj = edg.toJSON();
                    edgObj.put("sparql_query", goldenSparql);
                    edgObj.put("_id", serialNumber);
                    edgObj.put("intermediary_question", intermediateQuestion);
                    edgObj.put("sparql_template_id", template);

                    JSONArray candArray = new JSONArray();
                    int i = -1;
                    for (SparqlGenerator spg : sparqlList) {
                        i++;
                        JSONObject cand = new JSONObject();
                        cand.put("sparql_string", spg.toString());
                        cand.put("sparql_list", spg.getTupleList());
                        cand.put("sparql_type", spg.getQuesType());

                        IRMetrics metric = irMetrics[i];
                        cand.put("precision", metric.getPrecision());
                        cand.put("recall", metric.getRecall());
                        cand.put("f1", metric.getMicroF1());
                        candArray.put(cand);
                    }
                    edgObj.put("candidates", candArray);
                    array.put(edgObj);
                }
            } catch (Exception e) {
                e.printStackTrace();
            }

        }

        bfw1.write(array.toString(4));
        bfw1.flush();
        bfw1.close();
    }

    private static void computeMetrics(long startTime, CumulativeIRMetrics cumulativeIRMetrics, CumulativeIRMetrics QALDcumulativeIRMetrics, int quesIdx, long questionStartTime, int sparqlTemplateId, String goldenSparql, EDG edg, IRMetrics localIRMetrics, IRMetrics localQALDIRMetrics) {
        cumulativeIRMetrics.addSample(localIRMetrics);
        if (QAArgs.getDataset() == DatasetEnum.QALD_9) {
            QALDcumulativeIRMetrics.addSample(localQALDIRMetrics);
        }
        int curQuesNum = quesIdx + 1;

        LogUtil.printlnInfo("queryLog", "Cumulative metrics, " + cumulativeIRMetrics.toString());
        if (QAArgs.getDataset() == DatasetEnum.QALD_9) {
            LogUtil.printlnInfo("queryLog", "QALD Cumulative metrics, " + QALDcumulativeIRMetrics.toQALDString());
        }

        LogUtil.printlnInfo("queryLog", "Current question consumed time: " + (System.currentTimeMillis() - questionStartTime) + " ms, average: "
                + QAArgs.decimalFormat.format((double) (System.currentTimeMillis() - startTime) / curQuesNum)
                + " ms/ques, entity detection: " + QAArgs.decimalFormat.format((double) Timer.totalEntityLinkingTime / curQuesNum)
                + " ms/ques (linking tool: " + QAArgs.decimalFormat.format((double) Timer.totalEntityLinkingToolTime / curQuesNum)
                + " ms/ques), relation detection: " + QAArgs.decimalFormat.format((double) Timer.totalRelationLinkingTime / curQuesNum)
                + " ms/ques (similarity: " + QAArgs.decimalFormat.format((double) Timer.totalSimilarityTime / curQuesNum)
                + " ms/ques), EDG generation: " + QAArgs.decimalFormat.format((double) Timer.totalEDGGenTime / curQuesNum)
                + " ms/ques");  // timing
        LogUtil.printlnInfo("queryLog", Evaluator.getEntityLinkingMetricsStr());
        LogUtil.printlnInfo("queryLog", Evaluator.getRelationLinkingMetricsStr());
        LogUtil.printlnInfo("queryLog", Evaluator.getTypeLinkingMetricsStr());
        LogUtil.println("queryLog", "===============================================");
    }

    private static int getPredictedTripleNum(EDG edg) {
        int res;
        if (edg.getQueryType() == QueryType.JUDGE) {
            res = 1;
        } else res = edg.getNumDescriptiveNode();
        return res;
    }

    /**
     * Test each question manually
     * One question input, one answer set returned
     */
    public static void runQA() {
        while (true) {
            System.out.println("\n\nPlease enter the question: ");
            Scanner sc = new Scanner(System.in);
            String question = sc.nextLine();
            if (question.equals("quit()"))
                break;
            EDG edg = new EDG(question);
            System.out.println(edg);
            QuestionSolver questionSolver = new QuestionSolver(question, edg);
            System.out.println("Start solving question...");
            List<String> answers = questionSolver.solveQuestion();
            System.out.println("Predicted Sparql:" + questionSolver.getSubQuerySparqlMap().get(0));
            System.out.println("Predicted Answer:" + answers);
        }
    }

    /**
     * Test question by its id in the json
     */
    public static void runTestByQuestionId() {
        QAArgs.setCreatingLinkingCache(false);
        JSONArray sparqlArray = getDataArray();
        while (true) {
            System.out.println("\n\nPlease enter the question id: ");
            Scanner sc = new Scanner(System.in);
            int quesIdx = sc.nextInt();
            if (quesIdx < 0)
                break;
            CumulativeIRMetrics cumulativeIRMetrics = new CumulativeIRMetrics();
            CumulativeIRMetrics QALDcumulativeIRMetrics = new CumulativeIRMetrics();
            try {
                answerQuestions(System.currentTimeMillis(), cumulativeIRMetrics, QALDcumulativeIRMetrics, sparqlArray, quesIdx, quesIdx + 1, null);
            } catch (IOException e) {
                e.printStackTrace();
            }
            System.out.println("\n" + cumulativeIRMetrics.toString());
        }
    }

    /**
     * Test question by its serial number
     */
    public static void runTestBySerialNumber() {
        QAArgs.setCreatingLinkingCache(false);
        JSONArray sparqlArray = new JSONArray(FileUtil.readFileAsString("src/main/resources/datasets/lcquad-all.json"));
        while (true) {
            System.out.println("\n\nPlease enter the question serial number: ");
            Scanner sc = new Scanner(System.in);
            int serialNumber = sc.nextInt();
            if (serialNumber < 0)
                break;
            JSONObject jsonObject = null;
            for (int i = 0; i < sparqlArray.length(); i++) {
                JSONObject t = sparqlArray.getJSONObject(i);
                if (t.getInt("_id") == serialNumber) {
                    jsonObject = t;
                }
            }
            assert jsonObject != null;
            String question = jsonObject.getString("corrected_question");
            EDG edg = new EDG(question);
            System.out.println(edg);
            QuestionSolver questionSolver = new QuestionSolver(question, serialNumber, edg);
            System.out.println(questionSolver.solveQuestion());
        }
    }

    /**
     * initialize all the Classes
     *
     * @param dataset datasetToTest
     */
    public static void init(DatasetEnum dataset) {
        System.out.println("[INFO] Question Answering System initializing...");
        if (QAArgs.isCreatingLinkingCache())
            QAArgs.setUsingLinkingCache(false);
        QAArgs.setDataset(dataset);

        String set = QAArgs.isIsTraining() ? "train" : "test";
        System.out.println("[INFO] Dataset: " + QAArgs.getDatasetName() + ", " + set + " set");
        System.out.println("[INFO] Creating cache: " + QAArgs.isCreatingLinkingCache());
        System.out.println("[INFO] Using cache: " + QAArgs.isUsingLinkingCache());
        System.out.println("[INFO] Reranking: " + QAArgs.isReRanking());

        //EDG init
        EDG.init(dataset);
        //KBUtil init
        KBUtil.init(dataset);
        //Detector init
        Detector.init(dataset);

        if (QAArgs.getGoldenLinkingMode() == GoldenMode.FILTERING) {
            System.out.println("[INFO] Golden linking: filtering");
        } else if (QAArgs.getGoldenLinkingMode() == GoldenMode.DISABLED) {
            System.out.println("[INFO] Golden linking: disabled");
        } else if (QAArgs.getGoldenLinkingMode() == GoldenMode.GENERATION) {
            System.out.println("[INFO] Golden linking: generation");
        }
        System.out.println("[INFO] Question Solver initialized");
    }

    /**
     * Get the question type given a golden sparql
     *
     * @param goldenSparql the golden sparql
     * @return the QuestionType
     */
    public static QueryType getQueryTypeFromSparql(String goldenSparql) {
        if (goldenSparql.contains("COUNT") && goldenSparql.contains("SELECT"))
            return QueryType.COUNT;
        if (goldenSparql.contains("ASK"))
            return QueryType.JUDGE;
        return QueryType.COMMON;
    }

}
