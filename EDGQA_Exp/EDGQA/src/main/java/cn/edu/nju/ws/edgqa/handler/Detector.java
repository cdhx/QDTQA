package cn.edu.nju.ws.edgqa.handler;

import cn.edu.nju.ws.edgqa.domain.beans.Link;
import cn.edu.nju.ws.edgqa.domain.beans.LinkMap;
import cn.edu.nju.ws.edgqa.domain.beans.relation_detection.Paraphrase;
import cn.edu.nju.ws.edgqa.domain.edg.Node;
import cn.edu.nju.ws.edgqa.main.EDGQA;
import cn.edu.nju.ws.edgqa.main.QAArgs;
import cn.edu.nju.ws.edgqa.utils.Timer;
import cn.edu.nju.ws.edgqa.utils.*;
import cn.edu.nju.ws.edgqa.utils.enumerates.*;
import cn.edu.nju.ws.edgqa.utils.kbutil.KBUtil;
import cn.edu.nju.ws.edgqa.utils.linking.EntityLinkingThread;
import cn.edu.nju.ws.edgqa.utils.linking.GoldenLinkingUtil;
import cn.edu.nju.ws.edgqa.utils.linking.LinkingTool;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.io.*;
import java.util.*;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

import static cn.edu.nju.ws.edgqa.utils.CollectionUtil.setIntersect;

/**
 * The pipeline after EDG generation
 */
public class Detector {

    // entities covered of LC-Quad 1.0
    private static final Set<String> entityWhiteList = new HashSet<>();
    // predicates covered of LC-Quad 1.0
    private static final Set<String> predicateWhiteList = new HashSet<>(); // no-filtered property set
    private static final Set<String> predicateBlackList = new HashSet<>();
    // dbpedia 1604 classList
    private static final Set<String> dbpediaClass = new HashSet<>();
    // dbpedia-relation-paraphrases, key
    // : paraphrase words, value: paraphrase instance
    private static final Map<String, List<Paraphrase>> paraphraseMap = new HashMap<>(160000);
    // dbpedia ontologies for 201604, key: uri, value: label
    private static final HashMap<String, String> dbOntos = new HashMap<>();
    private static boolean whiteListFiltered = false; // by default, not to use whiteList
    // falcon stop-words
    private static List<String> stopWords;
    // auxiliary words for relation detection
    private static List<String> auxiliaryVerbs;

    /**
     * Initialize by dataset
     *
     * @param dataset the dataset id
     */
    public static void init(DatasetEnum dataset) {
        System.out.println("[INFO] Detector initializing...");

        //ontology in dbpedia 1604
        dbpediaClass.addAll(FileUtil.readFileAsList("src/main/resources/datasets/dbpedia1604Classes.txt"));

        //paraphrase dict
        List<Paraphrase> paraphraseList = new ArrayList<>();
        try {
            paraphraseList = FileUtil.loadParaphrase("src/main/resources/nlp/relation_paraphrase_dict.txt", "\t", true);
        } catch (IOException e) {
            e.printStackTrace();
        }
        buildParaphraseMap(paraphraseList);

        //stop words
        stopWords = FileUtil.readFileAsList("src/main/resources/nlp/stopwords-en.txt");
        //auxiliaryVerbs
        auxiliaryVerbs = FileUtil.readFileAsList("src/main/resources/nlp/auxiliary_verbs.txt");
        //dbpedia ontologies
        List<String> dbOntoUris = FileUtil.readFileAsList("src/main/resources/datasets/dbpedia1604Classes.txt");
        dbOntoUris.forEach(uri -> {
            String label = KBUtil.queryLabel(uri);
            if (label == null || label.trim().equals("")) {
                label = UriUtil.splitLabelFromUri(uri);
            }
            dbOntos.put(uri, label);
        });

        //whiteList and blackList
        if (dataset == DatasetEnum.LC_QUAD) {
            entityWhiteList.addAll(FileUtil.readFileAsList("src/main/resources/datasets/entity_whitelist_lcquad.txt"));
            predicateWhiteList.addAll(FileUtil.readFileAsList("src/main/resources/datasets/predicate_whitelist_lcquad.txt"));
            predicateBlackList.addAll(FileUtil.readFileAsList("src/main/resources/datasets/predicate_blacklist_lcquad.txt"));
        } else if (dataset == DatasetEnum.QALD_9) {
            predicateBlackList.addAll(FileUtil.readFileAsList("src/main/resources/datasets/predicate_blacklist_qald.txt"));
        }

        try {

            String directory = "cache/";
            if (QAArgs.isIsTraining()) {
                directory += "train/";
            } else {
                directory += "test/";
            }

            File dexterFile = new File(directory + "dexter_" + QAArgs.getDatasetName() + ".data");
            File earlFile = new File(directory + "earl_" + QAArgs.getDatasetName() + ".data");
            File falconFile = new File(directory + "falcon_" + QAArgs.getDatasetName() + ".data");

            if (QAArgs.isCreatingLinkingCache()) { // create linking cache
                CacheUtil.setDexterOutput(new ObjectOutputStream(new FileOutputStream(dexterFile)));
                CacheUtil.setEarlOutput(new ObjectOutputStream(new FileOutputStream(earlFile)));
                CacheUtil.setFalconOutput(new ObjectOutputStream(new FileOutputStream(falconFile)));
            } else { // read object from linking cache
                if (QAArgs.isUsingLinkingCache()) {
                    if (dexterFile.exists()) {
                        ObjectInputStream dexterInput = new ObjectInputStream(new FileInputStream(dexterFile));
                        CacheUtil.setDexterMap((Map<String, Map<String, List<Link>>>) dexterInput.readObject());
                    }
                    if (earlFile.exists()) {
                        ObjectInputStream earlInput = new ObjectInputStream(new FileInputStream(earlFile));
                        CacheUtil.setEarlMap((Map<String, Map<String, List<Link>>>) earlInput.readObject());
                    }
                    if (falconFile.exists()) {
                        ObjectInputStream falconInput = new ObjectInputStream(new FileInputStream(falconFile));
                        CacheUtil.setFalconMap((Map<String, Map<String, List<Link>>>) falconInput.readObject());
                    }
                }
            }

            File relationSimilarityFile = new File(directory + "relation_similarity_" + QAArgs.getDatasetName() + ".data");
            if (QAArgs.isCreatingRelationSimilarityCache()) { // create relation similarity cache
                CacheUtil.setRelationSimilarityOutput(new ObjectOutputStream(new FileOutputStream(relationSimilarityFile)));
            } else { // read relation similarity from cache
                if (QAArgs.isUsingRelationSimilarityCache() && relationSimilarityFile.exists()) {
                    ObjectInputStream relationSimilarityInput = new ObjectInputStream(new FileInputStream(relationSimilarityFile));
                    SimilarityUtil.setRelationSimilarityCache((Map<String, Double>) relationSimilarityInput.readObject());
                }
            }

        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }
        System.out.println("[INFO] Detector Initialization finished");
    }

    public static Set<String> getEntityWhiteList() {
        return entityWhiteList;
    }

    public static List<String> getAuxiliaryVerbs() {
        return auxiliaryVerbs;
    }

    public static Set<String> getPredicateWhiteList() {
        return predicateWhiteList;
    }

    public static Set<String> getPredicateBlackList() {
        return predicateBlackList;
    }

    public static Set<String> getDbpediaClass() {
        return dbpediaClass;
    }

    public static Map<String, List<Paraphrase>> getParaphraseMap() {
        return paraphraseMap;
    }

    public static List<String> getStopWords() {
        return stopWords;
    }

    public static boolean isWhiteListFiltered() {
        return whiteListFiltered;
    }

    public static void setWhiteListFiltered(boolean whiteListFiltered) {
        Detector.whiteListFiltered = whiteListFiltered;
    }

    /**
     * Build the paraphrase map based on the paraphrase list.
     *
     * @param paraphraseList the paraphrase list
     */
    public static void buildParaphraseMap(List<Paraphrase> paraphraseList) {
        for (Paraphrase paraphrase : paraphraseList) {
            if (!paraphraseMap.containsKey(paraphrase.getPredicate())) {
                paraphraseMap.put(paraphrase.getPredicate(), new ArrayList<>());
            }
            paraphraseMap.get(paraphrase.getPredicate()).add(paraphrase);
        }
    }

    /**
     * detect potential type constraint of a nodeStr
     *
     * @param sentence nodeStr
     * @return potential type List
     */
    @NotNull
    public static List<String> typeDetection(String sentence) {

        System.out.println("Detecting type:" + sentence);
        List<String> result = new ArrayList<>();

        for (String word : auxiliaryVerbs) {
            sentence = sentence.replace(word, "").trim();
        }

        // lemmatize of the sentence
        String lemmaSent = NLPUtil.getLemmaSent(sentence);

        /*List<String> tokens = NLPUtil.getTokens(sentence);
        // too long sentence
        if(tokens.size()>=5){
            return result;
        }*/

        for (String dbOnto : dbOntos.keySet()) {
            String dbOntoStr = UriUtil.splitLabelFromUri(dbOnto);
            String dbOntoLabel = dbOntos.get(dbOnto);
            if (SimilarityUtil.getScoreIgnoreCase(dbOntoStr, lemmaSent) > 0.8
                    || SimilarityUtil.getScoreIgnoreCase(dbOntoLabel, lemmaSent) > 0.8) {
                result.add(dbOnto);
            }
        }

        return result;

        /*List<String> typeLinkingList = TypeLinkingUtil.getTypeLinkingList(NLPUtil.getLemmaSent(sentence));
        System.out.println(sentence + " CL:" + typeLinkingList);
        return typeLinkingList;*/

    }

    /**
     * Entity Detection
     * Note that one node can detect at most one entity
     *
     * @param nodeStr             original sentence
     * @param globalEntityLinkMap link map for the whole sentence
     * @param goldenMode          golden linking mode, see GoldenLinkingUtil const
     * @param goldenLinking       the golden entity linking
     * @return sorted entity mention and the linking results
     */
    public static Map<String, List<Link>> entityDetection(@NotNull String nodeStr, Map<String, List<Link>> globalEntityLinkMap,
                                                          GoldenMode goldenMode, List<Link> goldenLinking) {
        long startTime = System.currentTimeMillis();
        System.out.println("\tEntityDetection:" + nodeStr);
        nodeStr = nodeStr.replace(" 's", "'s"); // Merge 'S 
        Map<String, List<Link>> resultMap = new ConcurrentHashMap<>();//The final entity detection result of the current NodeStr 

        long linkingToolTime = 0;
        if (QAArgs.isGoldenEntity() && goldenMode == GoldenMode.GENERATION) {
            // golden generation for entities
            Link link = GoldenLinkingUtil.getPotentialGoldenEntityLink(nodeStr, goldenLinking);
            List<Link> list = new ArrayList<>();
            list.add(link);
            if (link != null) {
                resultMap.put(link.getMention(), list);
            }
        } else {
            // the core function of entity detection
            linkingToolTime = entityDetectionCore(nodeStr, globalEntityLinkMap, goldenMode, goldenLinking, resultMap);
        }

        System.out.println("\tEntity detection resultMap:" + resultMap + "\n");
        // count entity linking time
        long timeCount = (System.currentTimeMillis() - startTime);
        synchronized (QuestionSolver.lock) {
            Timer.setTotalEntityLinkingTime(Timer.getTotalEntityLinkingTime() + timeCount);
            Timer.setTotalEntityLinkingToolTime(Timer.getTotalEntityLinkingToolTime() + linkingToolTime);
        }
        return resultMap;
    }

    private static long entityDetectionCore(@NotNull String nodeStr, Map<String, List<Link>> globalLinking, GoldenMode goldenMode,
                                            @Nullable List<Link> goldenLinking, @NotNull Map<String, List<Link>> resultMap) {

        long startLinkingToolTime = System.currentTimeMillis(); // timing for linking tools

        // Add the global entity linking results
        if (QAArgs.isGlobalEntityLinking()) {
            for (String key : globalLinking.keySet()) {
                if (key.contains(nodeStr)) {
                    resultMap.computeIfAbsent(nodeStr, k -> new ArrayList<>());
                    resultMap.get(nodeStr).addAll(globalLinking.get(key));
                }

                if (!QAArgs.isLocalEntityLinking()) {
                    if (nodeStr.contains(key)) {
                        resultMap.computeIfAbsent(nodeStr, k -> new ArrayList<>());
                        resultMap.get(nodeStr).addAll(globalLinking.get(key));
                    }
                }
            }
        }

        // Add the local entity linking results
        if (QAArgs.isLocalEntityLinking()) {
            try {

                // threads pool
                Callable<Map<String, List<Link>>> falconLinking = new EntityLinkingThread(EntityLinkingThread.LINKING_FALCON, nodeStr, resultMap);
                Future<Map<String, List<Link>>> falconRes = EDGQA.getPool().submit(falconLinking);
                Callable<Map<String, List<Link>>> dexterLinking = new EntityLinkingThread(EntityLinkingThread.LINKING_DEXTER, nodeStr, resultMap);
                Future<Map<String, List<Link>>> dexterRes = EDGQA.getPool().submit(dexterLinking);
                Callable<Map<String, List<Link>>> earlLinking = new EntityLinkingThread(EntityLinkingThread.LINKING_EARL, nodeStr, resultMap);
                Future<Map<String, List<Link>>> earlRes = EDGQA.getPool().submit(earlLinking);

                // Merge the linking results
                CollectionUtil.mergeLinkMap(LinkingTool.reScoreEntity(earlRes.get(QuestionSolver.entityLinkingThreadTimeLimit, TimeUnit.SECONDS)), resultMap, LinkEnum.ENTITY);
                CollectionUtil.mergeLinkMap(LinkingTool.reScoreEntity(dexterRes.get(QuestionSolver.entityLinkingThreadTimeLimit, TimeUnit.SECONDS)), resultMap, LinkEnum.ENTITY);
                CollectionUtil.mergeLinkMap(LinkingTool.reScoreEntity(falconRes.get(QuestionSolver.entityLinkingThreadTimeLimit, TimeUnit.SECONDS)), resultMap, LinkEnum.ENTITY);

                // remove the duplicate links and sort with reversed order
                resultMap.replaceAll((key, value) -> resultMap.get(key).stream().distinct().sorted(Collections.reverseOrder()).collect(Collectors.toList()));
                resultMap.forEach((key, value) -> value.removeIf(link -> link.getUri().endsWith("_(disambiguation)"))); // remove disambiguation page
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        long linkingToolTime = System.currentTimeMillis() - startLinkingToolTime;  // time spent on linking tools

        //Conflict span digestion 
        LinkingTool.spanConflictFix(nodeStr, resultMap);

        if (!QAArgs.isEvaluateCandNum()) {
            //Further reduce the results, using literally similarity 
            for (String key : resultMap.keySet()) {
                // remove the link if the score is lower than the threshold
                List<Link> linkList = resultMap.get(key);
                ListIterator<Link> linkIterator = linkList.listIterator();
                boolean equalFlag = false;
                while (linkIterator.hasNext()) {
                    Link link = linkIterator.next();

                    // The confidence is below the threshold, delete 
                    if (link.getScore() <= QuestionSolver.entityLinkThreshold) {
                        linkIterator.remove();
                        continue;
                    }

                    // DBPEDIA DISAMBIGUATIONPAGE, delete 
                    if (KBUtil.isDisambiguationPage(link.getUri())) {
                        linkIterator.remove();
                        continue;
                    }

                    // Is dbpedia redirectpage, modifying the entity after REDIRECT 
                    if (KBUtil.isRedirectPage(link.getUri())) {
                        List<String> redirectPages = KBUtil.getRedirectPage(link.getUri());
                        if (!redirectPages.isEmpty()) {
                            link.setUri(redirectPages.get(0));
                        }
                    }

                    //whether the entity label contains the whole entity mention
                    String linkMention = link.getMention();
                    String label = KBUtil.queryLabel(link.getUri());
                    if (label == null) {
                        label = UriUtil.splitLabelFromUri(link.getUri());
                    }
                    if (label.toLowerCase().contains(linkMention.toLowerCase().trim())) {//Contains full mout 
                        equalFlag = true;
                    }
                }

                //There is a completely identical link, and there is no completely containing Mention, removed. 
                if (equalFlag) {
                    Iterator<Link> iterator = linkList.iterator();
                    while (iterator.hasNext()) {
                        Link link = iterator.next();
                        String linkMention = link.getMention();
                        String label = KBUtil.queryLabel(link.getUri());
                        if (label == null) {
                            label = UriUtil.splitLabelFromUri(link.getUri());
                        }
                        if (!label.toLowerCase().contains(linkMention.toLowerCase())) {
                            iterator.remove();
                        }
                    }
                }

                // sort the linkList
                linkList.sort(Collections.reverseOrder());

                //Limit the number of Candidate Entity, limited to 3 
                /*while (linkList.size() > 3) {
                    linkList.remove(3);
                }*/
            }
        }

        // golden linking filtering
        if (QAArgs.isGoldenEntity()) {
            if (goldenMode != GoldenMode.DISABLED && (goldenLinking == null || goldenLinking.isEmpty())) {
                System.out.println("[INFO] golden entity linking set is empty");
            } else if (goldenMode == GoldenMode.FILTERING) {
                resultMap.forEach((k, v) -> v.removeIf(link -> goldenLinking.stream().noneMatch(golden -> golden.getUri().equals(link.getUri()))));
                resultMap.keySet().removeIf(key -> resultMap.get(key).isEmpty());
            }
        }

        return linkingToolTime;
    }

    /**
     * Relation detection
     *
     * @param nodeStr               original string of one EDG node
     * @param entityIndexMap        the entity id map used for removing entity mentions
     * @param eLinkList             the entity linking List
     * @param trigger               the relation Link result of the whole sentence by tools
     * @param goldenMode            golden linking mode, see GoldenLinkingUtil const
     * @param goldenLinking         the golden relation linking
     * @param globalRelationLinkMap the global relation LinkMap
     * @return sorted relation mention and the linking results
     */
    public static HashMap<String, List<Link>> relationDetection(@NotNull String nodeStr,
                                                                @NotNull Map<String, String> entityIndexMap,
                                                                List<Link> eLinkList,
                                                                Trigger trigger,
                                                                GoldenMode goldenMode,
                                                                @Nullable List<Link> goldenLinking,
                                                                LinkMap globalRelationLinkMap) {
        long startTime = System.currentTimeMillis();
        System.out.println("\tRelation Detection:" + nodeStr);

        HashMap<String, List<Link>> resultMap = new HashMap<>();
        HashSet<String> oneHopPropertySet = new HashSet<>();

        List<Link> likelyProperties = new ArrayList<>(); // Current Node possible Predicates 

        // process surface form
        String surface = nodeStr;
        // remove the entities to focus on the relation
        for (String entityId : entityIndexMap.keySet()) {
            surface = surface.replace(entityId, "");
        }
        // remove the auxiliary verbs
        for (String word : auxiliaryVerbs) {
            if (!word.startsWith("<"))
                word = " " + word + " "; // such as '<e0>', '<e1>'
            surface = surface.replace(word, " ");
        }
        surface = surface.replaceAll("[,./`'\"()]", "").replaceAll(" +", " ").trim();
        surface = NLPUtil.removeRedundantHeaderAndTailer(surface);
        surface = NLPUtil.getLemmaSent(surface);
        System.out.println("\tSurface:" + surface);


        if (surface.trim().equals("")) {// empty String, just return
            return resultMap;
        }

        if (QAArgs.isGoldenRelation() && goldenMode == GoldenMode.GENERATION) {
            Link link = GoldenLinkingUtil.getPotentialGoldenRelationLink(surface, goldenLinking);
            List<Link> list = new ArrayList<>();
            list.add(link);
            if (link != null) {
                resultMap.put(link.getMention(), list);
            }
        } else {

            // use global relation linking
            if (QAArgs.isGlobalRelationLinking() && globalRelationLinkMap != null) {
                for (String key : globalRelationLinkMap.getData().keySet()) {
                    if (key.contains(nodeStr)) {
                        resultMap.computeIfAbsent(nodeStr, k -> new ArrayList<>());
                        resultMap.get(nodeStr).addAll(globalRelationLinkMap.getData().get(key));
                    }

                    if (!QAArgs.isLocalRelationLinking()) {
                        if (nodeStr.contains(key)) {
                            resultMap.computeIfAbsent(nodeStr, k -> new ArrayList<>());
                            resultMap.get(nodeStr).addAll(globalRelationLinkMap.getData().get(key));
                        }
                    }

                }
            }

            // use localRelationLinking
            if (QAArgs.isLocalRelationLinking()) {
                relationDetectionCore(entityIndexMap, eLinkList, resultMap, oneHopPropertySet, likelyProperties, surface, trigger);
            }
        }

        // remove the stop-words
        //resultMap.entrySet().removeIf(entry -> stopWords.contains(entry.getKey()));

        // golden linking filtering
        if (QAArgs.isGoldenRelation()) {
            if (goldenMode != GoldenMode.DISABLED && (goldenLinking == null || goldenLinking.isEmpty())) {
                System.out.println("[INFO] golden relation linking set is empty");
            } else if (goldenMode == GoldenMode.FILTERING) {
                resultMap.forEach((k, v) -> v.removeIf(link ->
                        goldenLinking.stream().noneMatch(golden -> golden.getUri().equals(link.getUri()))));
                resultMap.keySet().removeIf(key -> resultMap.get(key).isEmpty());
            }
        }

        // only retain 5 candidate relation for each surface
        int relationBound = QAArgs.getRelationNumUpperB();
        if (QAArgs.isEvaluateCandNum()) {
            relationBound = 20;
        }

        int finalRelationBound = relationBound;
        resultMap.forEach((key, list) -> {
            while (list.size() > finalRelationBound) {
                list.remove(finalRelationBound);
            }
        });


        System.out.println("\tRelation Detection resultMap:" + resultMap);

        // count relation linking time
        long timeCount = System.currentTimeMillis() - startTime;
        synchronized (QuestionSolver.lock) {
            Timer.setTotalRelationLinkingTime(Timer.getTotalRelationLinkingTime() + timeCount);
        }
        return resultMap;
    }

    public static void relationDetectionCore(@NotNull Map<String, String> entityIndexMap,
                                             List<Link> eLinkList,
                                             HashMap<String, List<Link>> resultMap,
                                             HashSet<String> oneHopPropertySet,
                                             List<Link> likelyProperties,
                                             String surface, Trigger trigger) {
        // 1. we try to identify relations based on the identified entities without linking tools
        // 2. if there's no relations connected to the entities, we use relation linking tools


        // identify the relations based on the identified entities
        if (!eLinkList.isEmpty()) {// Entity Detection is not empty

            Iterator<Link> iter = eLinkList.iterator();
            while (iter.hasNext()) {
                Link link = iter.next();
                Set<String> filteredProperties = oneHopPropertyFiltered(link.getUri());
                // remove the type ontology
                filteredProperties.removeIf(url -> url.startsWith("http://dbpedia.org/ontology/") && Character.isUpperCase(url.charAt(28)));

                if (filteredProperties.isEmpty()) {
                    // remove the entity without valid relation directly
                    iter.remove();
                } else {
                    oneHopPropertySet.addAll(filteredProperties);
                }

            }

            //System.out.println("oneHopPropertySet:"+oneHopPropertySet);

            // calculate sim by set
            // key: writer ; value: [dbp:writer,dbr:writer,xxx:writer,...]
            Map<String, List<String>> labelUriMap = UriUtil.extractLabelMap(new ArrayList<>(oneHopPropertySet));
            //System.out.println("labelUriSize:"+labelUriMap.size());

            // calculate similarity by set
            HashMap<String, Double> labelSims = null;
            if (QAArgs.getTextMatchingModel() == TextMatching.COMPOSITE) {
                labelSims = SimilarityUtil.getCompositeSetSimilarity(surface, labelUriMap.keySet());
            } else if (QAArgs.getTextMatchingModel() == TextMatching.BERT) {
                labelSims = SimilarityUtil.getBertSetSimilarity(surface, labelUriMap.keySet());
            } else if (QAArgs.getTextMatchingModel() == TextMatching.LEXICAL) {
                labelSims = SimilarityUtil.getLexicalSetSimilarity(surface, labelUriMap.keySet());
            }

            if (labelSims != null) {
                for (String label : labelSims.keySet()) {
                    double score = labelSims.get(label);
                    if (score > QuestionSolver.relationLinkThreshold) { // threshold
                        for (String uri : labelUriMap.get(label)) {
                            likelyProperties.add(new Link(surface, uri, LinkEnum.RELATION, score));
                        }
                    }
                }
            }

            // resort the Relation List by reverse order
            LinkingTool.reSortRelationLinkList(likelyProperties);
            //System.out.println("Likely Properties:" + likelyProperties);

            //put to result Map
            resultMap.put(surface, likelyProperties);
        }

        // if no proper relation found according to entities, then we use relation linking tool
        if (likelyProperties.isEmpty()) {
            System.out.println("\t[DEBUG] No Relation Detected, Using tools");
            if (!surface.trim().equals("")) {

                // relation Linking by tools
                List<Link> candidateProps = LinkingTool.getRelationLinkingByTool(surface);

                // remove the type ontology
                candidateProps.removeIf(link -> link.getUri().startsWith("http://dbpedia.org/ontology/") && Character.isUpperCase(link.getUri().charAt(28)));

                // filter by black|white list
                if (whiteListFiltered)
                    candidateProps.removeIf(link -> !getPredicateWhiteList().contains(link.getUri()));
                else
                    candidateProps.removeIf(link -> getPredicateBlackList().contains(link.getUri()));

                // remove the duplicate
                candidateProps = candidateProps.stream().distinct().collect(Collectors.toList());

                // calculate score by set
                Map<String, List<String>> labelUriMap = UriUtil.extractLabelMap(candidateProps.stream().map(Link::getUri).collect(Collectors.toList()));

                //reset the candidate Props
                candidateProps = new ArrayList<>();
                HashMap<String, Double> labelSims = SimilarityUtil.getCompositeSetSimilarity(surface, labelUriMap.keySet());
                for (String label : labelSims.keySet()) {
                    double score = labelSims.get(label);
                    if (score > 0.1) {// beyond a threshold
                        for (String uri : labelUriMap.get(label)) {
                            candidateProps.add(new Link(surface, uri, LinkEnum.RELATION, score));
                        }
                    }
                }

                // remove the duplicate
                candidateProps = candidateProps.stream().distinct().collect(Collectors.toList());
                LinkingTool.reSortRelationLinkList(candidateProps);
                resultMap.put(surface, candidateProps);
            }
        }

        /* Use the trigger in the question to filter the relation links */
        if (trigger != Trigger.UNKNOWN) {
            for (Map.Entry<String, List<Link>> entry : resultMap.entrySet()) {
                for (Iterator<Link> iterator = entry.getValue().iterator(); iterator.hasNext(); ) {
                    Link link = iterator.next();
                    String uri = UriUtil.extractUri(link.getUri());
                    if (!filterLinkByTrigger(trigger, uri)) {
                        iterator.remove();
                    } else {
                        if (trigger == Trigger.WHEN) {
                            // score boost by trigger
                            link.setScore(Math.min(link.getScore() * 3, 1.0));
                        }
                    }
                }
            }
        }

        /* Remove candidate relations with large differences in scores */
        if (!QAArgs.isEvaluateCandNum()) {
            for (String key : resultMap.keySet()) {

                List<Link> rLinkList = resultMap.get(key);
                if (rLinkList.size() >= 2) {

                    Link firstLink = rLinkList.get(0);

                    // extremely high lexical similarity
                    String mention = firstLink.getMention();
                    String label = KBUtil.queryLabel(firstLink.getUri());
                    if (label == null) {
                        label = UriUtil.extractUri(firstLink.getUri());
                    }
                    if (SimilarityUtil.getScoreIgnoreCase(mention, label) > 0.9) {
                        while (rLinkList.size() > 1) {
                            rLinkList.remove(1);
                        }
                        continue;
                    }

                    // the relation label is explicitly contained by the surface form
                    if (mention.toLowerCase().contains(label.toLowerCase()) && label.length() >= 5) {
                        while (rLinkList.size() > 2) {
                            rLinkList.remove(2);
                        }

                        //Not the same label DBP / DBO, delete 
                        if (!UriUtil.extractUri(rLinkList.get(1).getUri()).equals(UriUtil.extractUri(firstLink.getUri()))) {
                            rLinkList.remove(1);
                        }
                        continue;
                    }


                    // relation score gap
                    for (int i = 0; i < rLinkList.size() - 1; i++) {
                        Link former = rLinkList.get(i);
                        Link latter = rLinkList.get(i + 1);

                        // find the gap
                        if (former.getScore() > 0.6 && latter.getScore() < 0.7 * former.getScore()) {
                            while (rLinkList.size() > i + 1) {
                                rLinkList.remove(i + 1);
                            }
                            break;
                        }
                    }
                }
            }
        }

    }

    public static boolean filterLinkByTrigger(Trigger trigger, String uri) {
        uri = uri.toLowerCase();
        if (trigger == Trigger.WHEN) {
            return uri.contains("date") || uri.contains("year") || uri.contains("month") || uri.contains("time");
        } else if (trigger == Trigger.WHERE) {
            return !uri.contains("date") && !uri.contains("time") && !uri.contains("cause") && !uri.contains("spouse");
        } else if (trigger == Trigger.WHO) {
            return !uri.contains("date") && !uri.contains("time") && !uri.contains("cause") && !uri.contains("place") && !uri.contains("year");
        } else if (trigger == Trigger.HOW) {
            return !uri.contains("place") && !uri.contains("time") && !uri.contains("date");
        }
        return true;
    }

    /**
     * Get the uri set of relations of one entity
     *
     * @param entityURI the uri of entity
     * @return the uri of adjacent relations
     */
    public static Set<String> oneHopPropertyFiltered(String entityURI) {
        Set<String> noFilteredProperties = KBUtil.oneHopProperty(entityURI);

        if (whiteListFiltered)
            return setIntersect(noFilteredProperties, predicateWhiteList);
        else {
            noFilteredProperties.removeAll(predicateBlackList);
        }

        if (QAArgs.getDataset() == DatasetEnum.LC_QUAD) {
            noFilteredProperties.removeIf(pred -> !pred.startsWith("http://dbpedia.org"));
        }
        return noFilteredProperties;
    }

    /**
     * Replace the Entity found in a node
     *
     * @param node      Need to change the Node
     * @param eLinkList Entity link result List
     * @return Map of entity ID and entity MENTION: {<E0>: Obama}
     */
    public static HashMap<String, String> replaceEntityInNode(Node node, List<Link> eLinkList) {
        //<e0,dbr:Obama>Such mapping 
        HashMap<String, String> entityIndexMap = new HashMap<>();
        //According to EntityLink results, the Node is converted 
        if (!eLinkList.isEmpty()) {
            int entityIdx = 0;
            Set<String> eMentionSet = eLinkList.stream().map(Link::getMention).collect(Collectors.toSet());

            for (String nodeEntityLink : eMentionSet) {
                int start = node.getStr().toLowerCase().indexOf(nodeEntityLink.toLowerCase()); // Avoid cases in cases 
                int end = start + nodeEntityLink.length();
                if (start != -1) {
                    String substring = node.getStr().substring(start, end);
                    node.setStr(node.getStr().replace(substring, "<e" + entityIdx + ">"));
                    entityIndexMap.put("<e" + entityIdx + ">", nodeEntityLink);
                } else {
                    //Not found in node.getstr () 
                    System.out.println("[DEBUG] entity mention: \"" + nodeEntityLink + "\" cannot be found, the node str is: " + node.getStr());
                }
                ++entityIdx;
            }

        } else {
            //No entity link result 
            System.out.println("\t[DEBUG] node " + node.getNodeID() + " entity link is empty");
        }
        return entityIndexMap;
    }
}