package cn.edu.nju.ws.edgqa.main;

import cn.edu.nju.ws.edgqa.handler.QuestionSolver;
import cn.edu.nju.ws.edgqa.utils.enumerates.DatasetEnum;
import cn.edu.nju.ws.edgqa.utils.enumerates.GoldenMode;
import cn.edu.nju.ws.edgqa.utils.enumerates.TextMatching;
import org.apache.commons.cli.*;

import java.text.DecimalFormat;

public class QAArgs {
    /**
     * The format of decimal
     */
    public static final DecimalFormat decimalFormat = new DecimalFormat("0.00");
    /**
     * Current dataset, -1 for default
     */
    private static DatasetEnum dataset = DatasetEnum.LC_QUAD;
    /**
     * training or set, true for training, false for test, false for default
     */
    private static boolean isTraining = false;
    /**
     * Whether to use golden linking to answer questions
     */
    private static GoldenMode goldenMode = GoldenMode.DISABLED;
    /**
     * Whether to use golden entity linking, if golden linking mode is enabled
     */
    private static boolean goldenEntity = false;
    /**
     * Whether to use golden relation linking, if golden linking mode is enabled
     */
    private static boolean goldenRelation = false;
    /**
     * Whether to use golden type linking, if golden linking mode is enabled
     */
    private static boolean goldenType = false;
    /**
     * Whether to build a file cache for entity linking
     */
    private static boolean creatingLinkingCache = false;
    /**
     * Whether to use linking cache
     */
    private static boolean usingLinkingCache = true;
    /**
     * Whether to use similarity cache
     */
    private static boolean usingRelationSimilarityCache = false;
    /**
     * whether to add type constraint
     */
    private static boolean usingTypeConstraints = true;
    /**
     * whether to create relation similarity cache
     */
    private static boolean creatingRelationSimilarityCache = false;

    /**
     * whether to use global entity linking
     */
    private static boolean globalEntityLinking = true;

    /**
     * whether to use global relation linking
     */
    private static boolean globalRelationLinking = true;

    /**
     * whether to use local entity linking
     */
    private static boolean localEntityLinking = true;

    /**
     * whether to use local relation linking
     */
    private static boolean localRelationLinking = true;

    /**
     * whether to use EDG to decompose the question
     */
    private static boolean questionDecomposition = true;

    /**
     * whether to evaluate the number of candidates
     */
    private static boolean evaluateCandNum = false;

    /**
     * whether to apply reranking
     */
    private static boolean reRanking = true;

//bao
    /**
     *which decomposition to run
     */
    public static String fileToRun = "";

    /**
     * whether to write logs
     */
    private static boolean writingLogs = true;

    /**
     * Relation ranking strategy
     */
    private static TextMatching textMatchingModel = TextMatching.COMPOSITE;
    /**
     * The number of candidate sparql queries for each entity block
     */
    private static int candidateSparqlNumUpperB = 2;
    /**
     * The number of candiate sparql queries for root entity
     */
    private static int rootCandidateSparqlNumUpperB = 1;

    /**
     * The upperbound of candidate sparql queries to rerank for each block
     */
    private static int blockSparqlNumUpperB = 5;

    /**
     * The upperbound of candidate relation for each description
     */
    private static int relationNumUpperB = 5;

    public static boolean isGlobalEntityLinking() {
        return globalEntityLinking;
    }

    public static void setGlobalEntityLinking(boolean globalEntityLinking) {
        QAArgs.globalEntityLinking = globalEntityLinking;
    }

    public static boolean isGlobalRelationLinking() {
        return globalRelationLinking;
    }

    public static void setGlobalRelationLinking(boolean globalRelationLinking) {
        QAArgs.globalRelationLinking = globalRelationLinking;
    }

    public static boolean isLocalEntityLinking() {
        return localEntityLinking;
    }

    public static void setLocalEntityLinking(boolean localEntityLinking) {
        QAArgs.localEntityLinking = localEntityLinking;
    }

    public static boolean isLocalRelationLinking() {
        return localRelationLinking;
    }

    public static void setLocalRelationLinking(boolean localRelationLinking) {
        QAArgs.localRelationLinking = localRelationLinking;
    }

    public static DatasetEnum getDataset() {
        return dataset;
    }

    public static void setDataset(DatasetEnum dataset) {
        QAArgs.dataset = dataset;
    }

    public static boolean isUsingTypeConstraints() {
        return usingTypeConstraints;
    }

    public static void setUsingTypeConstraints(boolean usingTypeConstraints) {
        QAArgs.usingTypeConstraints = usingTypeConstraints;
    }

    public static GoldenMode getGoldenLinkingMode() {
        return goldenMode;
    }

    public static void setGoldenLinkingMode(GoldenMode goldenMode) {
        QAArgs.goldenMode = goldenMode;
    }

    public static boolean isGoldenEntity() {
        return goldenEntity;
    }

    public static void setGoldenEntity(boolean goldenEntity) {
        QAArgs.goldenEntity = goldenEntity;
    }

    public static boolean isGoldenRelation() {
        return goldenRelation;
    }

    public static void setGoldenRelation(boolean goldenRelation) {
        QAArgs.goldenRelation = goldenRelation;
    }

    public static boolean isGoldenType() {
        return goldenType;
    }

    public static void setGoldenType(boolean goldenType) {
        QAArgs.goldenType = goldenType;
    }

    public static boolean isCreatingLinkingCache() {
        return creatingLinkingCache;
    }

    public static void setCreatingLinkingCache(boolean creatingLinkingCache) {
        QAArgs.creatingLinkingCache = creatingLinkingCache;
    }

    public static boolean isUsingLinkingCache() {
        return usingLinkingCache;
    }

    public static void setUsingLinkingCache(boolean usingLinkingCache) {
        QAArgs.usingLinkingCache = usingLinkingCache;
    }

    public static boolean isCreatingRelationSimilarityCache() {
        return QAArgs.creatingRelationSimilarityCache;
    }

    public static void setCreatingRelationSimilarityCache(boolean creatingRelationSimilarityCache) {
        QAArgs.creatingRelationSimilarityCache = creatingRelationSimilarityCache;
    }

    public static boolean isUsingRelationSimilarityCache() {
        return QAArgs.usingRelationSimilarityCache;
    }

    public static void setUsingRelationSimilarityCache(boolean usingRelationSimilarityCache) {
        QAArgs.usingRelationSimilarityCache = usingRelationSimilarityCache;
    }

    public static TextMatching getTextMatchingModel() {
        return QAArgs.textMatchingModel;
    }

    public static void setTextMatchingModel(TextMatching textMatchingModel) {
        QAArgs.textMatchingModel = textMatchingModel;
    }

    public static String getDatasetName() {
        if (dataset == DatasetEnum.LC_QUAD)
            return "lc-quad";
        else if (dataset == DatasetEnum.QALD_9)
            return "qald-9";
        return null;
    }

    public static int getCandidateSparqlNumUpperB() {
        return candidateSparqlNumUpperB;
    }

    public static void setCandidateSparqlNumUpperB(int candidateSparqlNumUpperB) {
        QAArgs.candidateSparqlNumUpperB = candidateSparqlNumUpperB;
    }

    public static int getRootCandidateSparqlNumUpperB() {
        return rootCandidateSparqlNumUpperB;
    }

    public static void setRootCandidateSparqlNumUpperB(int rootCandidateSparqlNumUpperB) {
        QAArgs.rootCandidateSparqlNumUpperB = rootCandidateSparqlNumUpperB;
    }

    public static boolean isWritingLogs() {
        return writingLogs;
    }

    public static void setWritingLogs(boolean writingLogs) {
        QAArgs.writingLogs = writingLogs;
    }


    public static boolean isQuestionDecomposition() {
        return questionDecomposition;
    }

    public static void setQuestionDecomposition(boolean questionDecomposition) {
        QAArgs.questionDecomposition = questionDecomposition;
    }

    public static boolean isEvaluateCandNum() {
        return evaluateCandNum;
    }

    public static void setEvaluateCandNum(boolean evaluateCandNum) {
        QAArgs.evaluateCandNum = evaluateCandNum;
    }

    public static boolean isIsTraining() {
        return isTraining;
    }

    public static void setIsTraining(boolean isTraining) {
        QAArgs.isTraining = isTraining;
    }

    public static boolean isReRanking() {
        return reRanking;
    }

    public static void setReRanking(boolean reRanking) {
        QAArgs.reRanking = reRanking;
    }

    public static void setFileToRun(String file) {
        QAArgs.fileToRun = file;
    }

    public static int getBlockSparqlNumUpperB() {
        return blockSparqlNumUpperB;
    }

    public static void setBlockSparqlNumUpperB(int blockSparqlNumUpperB) {
        QAArgs.blockSparqlNumUpperB = blockSparqlNumUpperB;
    }

    public static int getRelationNumUpperB() {
        return relationNumUpperB;
    }

    public static void setRelationNumUpperB(int relationNumUpperB) {
        QAArgs.relationNumUpperB = relationNumUpperB;
    }

    public static Options getOptions() {
        Options options = new Options();
        Option dataset = new Option("d", "dataset", true, "the QA dataset, 'qald-7', 'qald-9', or 'lc-quad'");
        dataset.setRequired(true);
        options.addOption(dataset);

        Option run = new Option("r", "run", true, "running mode, 'autotest', 'single', or 'serial_number' ");
        run.setRequired(false);
        options.addOption(run);

        Option train = new Option("tr", "train", true, "train set or test set");
        run.setRequired(false);
        options.addOption(train);

        Option goldenAnswer = new Option("ga", "golden_answer", true, "the way of using golden linking to answer questions: 'disabled', 'filtering', 'generation'");
        goldenAnswer.setRequired(false);
        options.addOption(goldenAnswer);

        Option goldenEntity = new Option("ge", "golden_entity", true, "whether to use golden entity linking, 'true' or 'false'");
        goldenEntity.setRequired(false);
        options.addOption(goldenEntity);

        Option goldenRelation = new Option("gr", "golden_relation", true, "whether to use golden relation linking, 'true' or 'false'");
        goldenRelation.setRequired(false);
        options.addOption(goldenRelation);

        Option goldenType = new Option("gt", "golden_type", true, "whether to use golden type linking, 'true' or 'false'");
        goldenType.setRequired(false);
        options.addOption(goldenType);

        Option createLinkingCache = new Option("cc", "create_cache", true, "whether to create file cache for global linking");
        createLinkingCache.setRequired(false);
        options.addOption(createLinkingCache);

        Option useLinkingCache = new Option("uc", "use_cache", true, "whether to use file cache for golden linking");
        useLinkingCache.setRequired(false);
        options.addOption(useLinkingCache);

        Option entityLinkingThreadTimeLimit = new Option("elt", "el_time", true, "the time limit for each entity linking thread");
        entityLinkingThreadTimeLimit.setRequired(false);
        options.addOption(entityLinkingThreadTimeLimit);

        Option candidateSparqlNumberUpperB = new Option("sn", "sparql_number", true, "the number of candidate sparql queries for each entity block");
        candidateSparqlNumberUpperB.setRequired(false);
        options.addOption(candidateSparqlNumberUpperB);

        Option rootCandidateSparqlNumUpperB = new Option("rsn", "root_sparql_number", true, "the number of candidate sparql queries for root entity");
        rootCandidateSparqlNumUpperB.setRequired(false);
        options.addOption(rootCandidateSparqlNumUpperB);

        Option useRelationSimilarityCache = new Option("ursc", "use_relation_similarity_cache", true, "whether to use relation similarity cache");
        useRelationSimilarityCache.setRequired(false);
        options.addOption(useRelationSimilarityCache);

        Option createRelationSimilarityCache = new Option("crsc", "create_relation_similarity_cache", true, "whether to create relation similarity cache");
        createRelationSimilarityCache.setRequired(false);
        options.addOption(createRelationSimilarityCache);

        Option textMatching = new Option("tm", "text_matching", true, "text matching model: 'bert', 'literal', or 'composite'");
        textMatching.setRequired(false);
        options.addOption(textMatching);

        Option typeConstraints = new Option("tc", "type_constraints", true, "whether to use type constraints");
        typeConstraints.setRequired(false);
        options.addOption(typeConstraints);

        Option globalLinking = new Option("gll", "global_linking", true, "whether to use global linking");
        globalLinking.setRequired(false);
        options.addOption(globalLinking);

        Option localLinking = new Option("lll", "local_linking", true, "whether to use local linking");
        localLinking.setRequired(false);
        options.addOption(localLinking);

        Option questionDecomposition = new Option("qd", "question_decomposition", true, "whether to use EDG to decompose the question");
        questionDecomposition.setRequired(false);
        options.addOption(questionDecomposition);

        Option evaluateCandNum = new Option("ec", "evaluate_candidate_num", true, "evaluate the number of candidates");
        evaluateCandNum.setRequired(false);
        options.addOption(evaluateCandNum);

        Option log = new Option("log", "log", true, "whether to write file logs");
        log.setRequired(false);
        options.addOption(log);

        Option reranking = new Option("rr", "reranking", true, "whether to rerank");
        reranking.setRequired(false);
        options.addOption(reranking);

        Option filetorun = new Option("file", "file", true, "file to run");
        filetorun.setRequired(false);
        options.addOption(filetorun);

        return options;
    }

    public static int setArguments(Options options, String[] args) {
        CommandLineParser parser = new DefaultParser();
        CommandLine commandLine = null;
        try {
            commandLine = parser.parse(options, args);
        } catch (ParseException e) {
            e.printStackTrace();
            System.out.println("[ERROR] arguments parsing exception");
        }

        assert commandLine != null;
        if (commandLine.hasOption("train")) {
            switch (commandLine.getOptionValues("train")[0].toLowerCase()) {
                case "true":
                    setIsTraining(true);
                    break;
                case "false":
                    setIsTraining(false);
                    break;
            }
        }
        if (commandLine.hasOption("golden_answer")) {
            switch (commandLine.getOptionValues("golden_answer")[0]) {
                case "disabled":
                    setGoldenLinkingMode(GoldenMode.DISABLED);
                    break;
                case "filtering":
                    setGoldenLinkingMode(GoldenMode.FILTERING);
                    break;
                case "generation":
                    setGoldenLinkingMode(GoldenMode.GENERATION);
                    break;
            }
        }
        if (commandLine.hasOption("golden_entity")) {
            switch (commandLine.getOptionValues("golden_entity")[0].toLowerCase()) {
                case "true":
                    setGoldenEntity(true);
                    break;
                case "false":
                    setGoldenEntity(false);
                    break;
            }
        }
        if (commandLine.hasOption("golden_relation")) {
            switch (commandLine.getOptionValues("golden_relation")[0].toLowerCase()) {
                case "true":
                    setGoldenRelation(true);
                    break;
                case "false":
                    setGoldenRelation(false);
                    break;
            }
        }
        if (commandLine.hasOption("golden_type")) {
            switch (commandLine.getOptionValues("golden_type")[0].toLowerCase()) {
                case "true":
                    setGoldenType(true);
                    break;
                case "false":
                    setGoldenType(false);
                    break;
            }
        }
        if (commandLine.hasOption("create_cache")) {
            String create_global_cache = commandLine.getOptionValues("create_cache")[0].toLowerCase();
            setCreatingLinkingCache("true".equals(create_global_cache));
        }
        if (commandLine.hasOption("use_cache")) {
            String useLinkingCache = commandLine.getOptionValues("use_cache")[0].toLowerCase();
            setUsingLinkingCache("true".equals(useLinkingCache));
        }
        if (commandLine.hasOption("el_time")) {
            QuestionSolver.setEntityLinkingThreadTimeLimit(Integer.parseInt(commandLine.getOptionValues("el_time")[0]));
        }
        if (commandLine.hasOption("sparql_number")) {
            setCandidateSparqlNumUpperB(Integer.parseInt(commandLine.getOptionValues("sparql_number")[0]));
        }
        if (commandLine.hasOption("root_sparql_number")) {
            setRootCandidateSparqlNumUpperB(Integer.parseInt(commandLine.getOptionValues("root_sparql_number")[0]));
        }
        if (commandLine.hasOption("use_relation_similarity_cache")) {
            setUsingRelationSimilarityCache(Boolean.parseBoolean(commandLine.getOptionValues("use_relation_similarity_cache")[0]));
        }
        if (commandLine.hasOption("create_relation_similarity_cache")) {
            setCreatingRelationSimilarityCache(Boolean.parseBoolean(commandLine.getOptionValues("create_relation_similarity_cache")[0]));
        }

        if (commandLine.hasOption("text_matching")) {
            switch (commandLine.getOptionValues("text_matching")[0]) {
                case "bert":
                    setTextMatchingModel(TextMatching.BERT);
                    break;
                case "literal":
                    setTextMatchingModel(TextMatching.LEXICAL);
                    break;
                default:
                    setTextMatchingModel(TextMatching.COMPOSITE);
                    break;
            }
        }

        if (commandLine.hasOption("type_constraints")) {
            String typeConstraintsArg = commandLine.getOptionValues("type_constraints")[0].toLowerCase();
            if ("true".equals(typeConstraintsArg)) {
                setUsingTypeConstraints(true);
            } else if ("false".equals(typeConstraintsArg)) {
                setUsingTypeConstraints(false);
            }
        }

        if (commandLine.hasOption("global_linking")) {
            String globalLinkingArg = commandLine.getOptionValues("global_linking")[0].toLowerCase();
            if ("true".equals(globalLinkingArg)) {
                setGlobalEntityLinking(true);
                setGlobalRelationLinking(true);
            } else if ("false".equals(globalLinkingArg)) {
                setGlobalEntityLinking(false);
                setGlobalRelationLinking(false);
            }
        }

        if (commandLine.hasOption("local_linking")) {
            String localLinkingArg = commandLine.getOptionValues("local_linking")[0].toLowerCase();
            if ("true".equals(localLinkingArg)) {
                setLocalEntityLinking(true);
                setLocalRelationLinking(true);
            } else if ("false".equals(localLinkingArg)) {
                setLocalEntityLinking(false);
                setLocalRelationLinking(false);
            }
        }

        if (commandLine.hasOption("question_decomposition")) {
            String questionDecompositionArg = commandLine.getOptionValues("question_decomposition")[0].toLowerCase();
            if ("true".equals(questionDecompositionArg)) {
                setQuestionDecomposition(true);
            } else if ("false".equals(questionDecompositionArg)) {
                setQuestionDecomposition(false);
            }
        }

        if (commandLine.hasOption("evaluate_candidate_num")) {
            String evaluateCandNumArg = commandLine.getOptionValues("evaluate_candidate_num")[0].toLowerCase();
            if ("true".equals(evaluateCandNumArg)) {
                setEvaluateCandNum(true);
            } else if ("false".equals(evaluateCandNumArg)) {
                setEvaluateCandNum(false);
            }
        }

        if (commandLine.hasOption("reranking")) {
            String rerankingArg = commandLine.getOptionValues("reranking")[0].toLowerCase();
            if ("true".equals(rerankingArg)) {
                setReRanking(true);
            } else if ("false".equals(rerankingArg)) {
                setReRanking(false);
            }
        }

        if (commandLine.hasOption("file")) {
            String fileArg = commandLine.getOptionValues("file")[0];
            {
                setFileToRun(fileArg);
            }
        }

        if (commandLine.hasOption("log")) {
            String logArg = commandLine.getOptionValues("log")[0].toLowerCase();
            if ("true".equals(logArg)) {
                setWritingLogs(true);
            } else if ("false".equals(logArg)) {
                setWritingLogs(false);
            }
        }

        switch (commandLine.getOptionValues("dataset")[0]) {
            case "lc-quad":
                EDGQA.init(DatasetEnum.LC_QUAD);
                break;
            case "qald-9":
                EDGQA.init(DatasetEnum.QALD_9);
                break;
        }

        if (commandLine.hasOption("run")) {
            switch (commandLine.getOptionValues("run")[0]) {
                case "autotest":
                    return 0;
                case "single":
                    return 1;
                case "serial_number":
                    return 2;
                case "question_id":
                    return 3;
                case "cand_evaluate":
                    return 4;
            }
        }
        return 0;
    }
}
