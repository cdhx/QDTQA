package cn.edu.nju.ws.edgqa.utils.linking;

import cn.edu.nju.ws.edgqa.domain.beans.Link;
import cn.edu.nju.ws.edgqa.domain.beans.Span;
import cn.edu.nju.ws.edgqa.handler.Detector;
import cn.edu.nju.ws.edgqa.utils.CacheUtil;
import cn.edu.nju.ws.edgqa.main.QAArgs;
import cn.edu.nju.ws.edgqa.utils.SimilarityUtil;
import cn.edu.nju.ws.edgqa.utils.UriUtil;
import cn.edu.nju.ws.edgqa.utils.connect.HttpsClientUtil;
import cn.edu.nju.ws.edgqa.utils.enumerates.KBEnum;
import cn.edu.nju.ws.edgqa.utils.enumerates.LinkEnum;
import cn.edu.nju.ws.edgqa.utils.enumerates.ToolEnum;
import cn.edu.nju.ws.edgqa.utils.kbutil.KBUtil;
import org.apache.jena.query.QueryFactory;
import org.apache.jena.query.QuerySolution;
import org.json.JSONArray;
import org.json.JSONObject;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;


public class LinkingTool {

    //dexter server IP and port
    public static final String dexterServerIP = "114.212.190.19";
    public static final String dexterLocalUrl = "http://" + dexterServerIP + ":8080/dexter-webapp/api/rest/spot";

    // earl server IP and port
    private static final String earlServerIP = "114.212.190.19";
    private static final int earlServerPort = 4999;
    private static final String earlLocalURL = "http://" + earlServerIP + ":" + earlServerPort + "/processQuery";
    private static final String earlLiveURL = "http://ltdemos.informatik.uni-hamburg.de/earl/processQuery";

    // falcon server IP and portF
    private static final String falconServerlIP = "114.212.190.19";
    private static final int falconServerPort = 9876;
    private static final String falconLocalURL = "http://" + falconServerlIP + ":" + falconServerPort + "/annotate?k=";
    private static final String falconLiveURL = "https://labs.tib.eu/falcon/api?mode=long&k=";

    private static final String earlURL = earlLocalURL;
    private static final String falconURL = falconLocalURL;

    /**
     * Enter a sentence and return the result Map of EARL Linking (including relations and entities at the same time)
     *
     * @param sentence natural language question
     * @return EARL link result map
     */
    public static Map<String, List<Link>> getEARLLinking(String sentence) {
        if (QAArgs.isUsingLinkingCache() && CacheUtil.getEarlMap() != null && CacheUtil.getEarlMap().containsKey(sentence))
            return CacheUtil.getEarlMap().get(sentence);

        Map<String, List<Link>> result = new ConcurrentHashMap<>();
        boolean pageRankFlag = false;

        JSONObject jsonObject = new JSONObject();
        jsonObject.put("nlquery", sentence);
        jsonObject.put("pagerankflag", pageRankFlag);

        int tryTime = 0;
        while (tryTime < 2 && result.size() == 0) {
            try {

                String jsonString = HttpsClientUtil.doPost(earlURL, jsonObject.toString());
                //System.out.println(jsonString);
                if (jsonString != null) {
                    JSONObject o1 = new JSONObject(jsonString);
                    JSONArray types = (JSONArray) o1.get("ertypes");

                    int num = types.length();

                    JSONObject reRankedList = (JSONObject) o1.get("rerankedlists");
                    JSONArray chunkText = o1.getJSONArray("chunktext");

                    for (int i = 0; i < num; i++) {
                        //System.out.println(rerankedList.get(String.valueOf(i)).getClass());
                        boolean isEntity = types.getString(i).equals("entity");
                        JSONArray linkArray = (JSONArray) reRankedList.get(String.valueOf(i));
                        JSONObject chunkObject = chunkText.getJSONObject(i);
                        String chunk = chunkObject.getString("chunk");
                        List<Link> list = new ArrayList<>();
                        for (int j = 0; j < linkArray.length(); j++) {
                            JSONArray link = (JSONArray) linkArray.get(j);
                            double rank = link.getDouble(0);
                            String linkUri = link.getString(1);
                            //list.add(linkUri);
                            list.add(new Link(chunk, linkUri, isEntity ? LinkEnum.ENTITY : LinkEnum.RELATION, rank));
                        }
                        result.put(chunk, list);

                        //System.out.println(rerankedList.get(String.valueOf(i)));
                    }

                }
            } catch (Exception e) {
                e.printStackTrace();
            } finally {
                tryTime++;

            }
        }
        if (QAArgs.isCreatingLinkingCache())
            CacheUtil.getEarlMap().put(sentence, result);
        return result;
    }

    /**
     * Enter a sentence and return the result Map of falcon Linking (including relations and entities at the same time). By default, each Mention returns 10 candidates
     *
     * @param sentence natural language question
     * @return Link result map
     */
    public static Map<String, List<Link>> getFalconLinking(String sentence) {
        if (QAArgs.isUsingLinkingCache() && CacheUtil.getFalconMap() != null && CacheUtil.getFalconMap().containsKey(sentence))
            return CacheUtil.getFalconMap().get(sentence);

        Map<String, List<Link>> result = new ConcurrentHashMap<>();
        int tryTime = 0;
        //boolean exceptionFlag = true;
        while (tryTime < 2 && result.size() == 0) {
            //exceptionFlag=false;
            JSONObject jsonObject = new JSONObject();
            jsonObject.put("text", sentence);
            int k = 10;
            jsonObject.put("k", k);
            try {

                String url = falconURL + k;
                String response = HttpsClientUtil.doPost(url, jsonObject.toString());
                //System.out.println(response);
                if (response != null) {
                    JSONObject o1 = new JSONObject(response);
                    //System.out.println(o1.toString(4));

                    JSONArray dbpediaEntities = o1.getJSONArray("entities");

                    double intialProb = 0.05;
                    for (int i = 0; i < dbpediaEntities.length(); i++) {
                        String uri = dbpediaEntities.getJSONArray(i).getString(0);
                        String surface = dbpediaEntities.getJSONArray(i).getString(1);
                        Link curLink = new Link(surface, uri, LinkEnum.ENTITY, intialProb);
                        intialProb *= 0.5;
                        if (result.containsKey(surface)) {

                            List<Link> list = result.get(surface);
                            list.add(curLink);
                        } else {
                            ArrayList<Link> list = new ArrayList<>();
                            list.add(curLink);
                            result.put(surface, list);
                        }


                    }
                    //System.out.println(HTTPUtil.sendPost(url,jsonObject.toString()));

                    JSONArray dbpediaRelations = o1.getJSONArray("relations");
                    intialProb = 0.05;
                    for (int i = 0; i < dbpediaRelations.length(); i++) {
                        String uri = dbpediaRelations.getJSONArray(i).getString(0);
                        String surface = dbpediaRelations.getJSONArray(i).getString(1);
                        Link curLink = new Link(surface, uri, LinkEnum.RELATION, intialProb);
                        intialProb *= 0.5;
                        if (result.containsKey(surface)) {
                            List<Link> list = result.get(surface);
                            list.add(curLink);
                        } else {
                            ArrayList<Link> list = new ArrayList<>();
                            list.add(curLink);
                            result.put(surface, list);
                        }
                    }

                }
            } catch (Exception e) {
                //e.printStackTrace();
                //exceptionFlag=true;
            } finally {
                tryTime++;
            }
        }
        if (QAArgs.isCreatingLinkingCache())
            CacheUtil.getFalconMap().put(sentence, result);
        return result;
    }


    /**
     * Enter a sentence and return the dexter entity linking result Map
     *
     * @param sentence natural language question
     * @return dexter entity linking Map
     */
    public static Map<String, List<Link>> getDexterLinking(String sentence) {
        if (QAArgs.isUsingLinkingCache() && CacheUtil.getDexterMap() != null && CacheUtil.getDexterMap().containsKey(sentence))
            return CacheUtil.getDexterMap().get(sentence);

        Map<String, List<Integer>> candidateEntityIDs = getCandidateEntityIDs_dexter(sentence);
        //System.out.println("Candidate ID:"+candidateEntityIDs);
        Map<String, List<Link>> result = new ConcurrentHashMap<>();
        for (String key : candidateEntityIDs.keySet()) {
            List<Integer> ids = candidateEntityIDs.get(key);
            ArrayList<Link> linkList = new ArrayList<>();
            for (int i : ids) {
                String s = KBUtil.queryByID(i);
                if (s != null && !s.equals("")) {
                    linkList.add(new Link(key, s, LinkEnum.ENTITY));
                }
            }
            result.put(key, linkList);
        }

        if (QAArgs.isCreatingLinkingCache())
            CacheUtil.getDexterMap().put(sentence, result);
        return result;
    }

    /**
     * Enter a mention, return a list of all possible relationship link results
     *
     * @param mention mention
     * @return relation linking List
     */
    public static List<Link> getRelationLinkingByTool(String mention) {
        List<Link> candidateProps = new ArrayList<>();

        if (mention.trim().equals("") || mention.trim().equals("[,.?!`_\\-\\d]")) {
            return candidateProps;
        }

        Map<String, List<Link>> earlLinking = LinkingTool.getEARLLinking(mention);
        Map<String, List<Link>> falconLinking = LinkingTool.getFalconLinking(mention);

        for (String key : earlLinking.keySet()) {
            if (earlLinking.get(key).get(0).getType() == LinkEnum.RELATION) {
                candidateProps.addAll(earlLinking.get(key));
            }
        }
        for (String key : falconLinking.keySet()) {
            if (falconLinking.get(key).get(0).getType() == LinkEnum.RELATION) {
                candidateProps.addAll(falconLinking.get(key));
            }
        }
        return candidateProps;
    }

    /**
     * Enter a sentence and return the RL\EL result generated by tool to RMAP\EMAP
     *
     * @param sentence the whole sentence
     * @param eLinkMap the entity linking map
     * @param rLinkMap the relation linking map
     * @param tool     the linking tool
     */
    public static void getToolLinkingMaps(String sentence, Map<String, List<Link>> eLinkMap, Map<String, List<Link>> rLinkMap, ToolEnum tool) {
        Map<String, List<Link>> toolLinkingMap;
        if (tool == ToolEnum.EARL) {
            toolLinkingMap = getEARLLinking(sentence);
        } else if (tool == ToolEnum.FALCON) {
            toolLinkingMap = getFalconLinking(sentence);
        } else if (tool == ToolEnum.DEXTER) {
            toolLinkingMap = getDexterLinking(sentence);
        } else {
            System.out.println("[Error] Invalid tool");
            return;
        }

        // check null pointer
        if (eLinkMap == null) eLinkMap = new HashMap<>();
        if (rLinkMap == null) rLinkMap = new HashMap<>();

        for (String key : toolLinkingMap.keySet()) {
            List<Link> links = toolLinkingMap.get(key);

            if (!links.isEmpty()) { // not empty
                Map<String, List<Link>> targetMap = rLinkMap; // default rLinkMap
                if (links.get(0).getType() == LinkEnum.ENTITY) {  // eLinkMap
                    targetMap = eLinkMap;
                }
                if (targetMap.containsKey(key)) {
                    targetMap.get(key).addAll(links);
                } else {
                    targetMap.put(key, new ArrayList<>(links));
                }
            }
        }
    }

    /**
     * Enter a sentence and return the link result after the fusion of multiple tools
     *
     * @param sentence  question to link
     * @param eLinkMap  entity linking MAP
     * @param rLinkMap  relation linking MAP
     * @param useFalcon whether to use linking tool Falcon
     */
    public static void getEnsembleLinking(String sentence, Map<String, List<Link>> eLinkMap, Map<String, List<Link>> rLinkMap, boolean useFalcon) {

        getToolLinkingMaps(sentence, eLinkMap, rLinkMap, ToolEnum.DEXTER);
        getToolLinkingMaps(sentence, eLinkMap, rLinkMap, ToolEnum.EARL);
        if (useFalcon) {
            getToolLinkingMaps(sentence, eLinkMap, rLinkMap, ToolEnum.FALCON);
        }

        //remove duplicated linking result
        eLinkMap.replaceAll((k, v) -> eLinkMap.get(k).stream().distinct().collect(Collectors.toList()));
        rLinkMap.replaceAll((k, v) -> rLinkMap.get(k).stream().distinct().collect(Collectors.toList()));

        //rescore entity
        reScoreEntity(eLinkMap);

        //entity Mention Conflict resolution
        spanConflictFix(sentence, eLinkMap);

        //rescore relation
        reScoreRelation(rLinkMap);
    }

    /**
     * Resolve conflicting mentions in eLinkMap
     *
     * @param sentence original sentence
     * @param eLinkMap mention to solve
     */
    public static Map<String, List<Link>> spanConflictFix(String sentence, Map<String, List<Link>> eLinkMap) {
        //Possible entity spans considered by several detection tools
        List<Span> spans = new ArrayList<>();
        for (String mention : eLinkMap.keySet()) {
            int start = sentence.toLowerCase().indexOf(mention.toLowerCase());
            int end = start + mention.length();
            spans.add(new Span(start, end, mention));
        }

        //More than two entities, need to remove the overlapping mention
        if (spans.size() >= 2) {
            //The span to be deleted, removeFlag[i]=true indicates that the span at position i should be deleted
            boolean[] removeFlag = new boolean[spans.size()]; //all false by default
            for (int i = 0; i < spans.size(); i++) {
                if (removeFlag[i]) {//span to remove, continue
                    continue;
                }
                Span span1 = spans.get(i);
                for (int j = i + 1; j < spans.size(); j++) {
                    if (removeFlag[j]) {//span to remove, continue
                        continue;
                    }
                    Span span2 = spans.get(j);
                    if (Span.spanConflict(span1, span2)) {  // judge if two spans have overlap
                        double conf1 = SimilarityUtil.getMentionConfidence(span1.getStr(), eLinkMap);
                        double conf2 = SimilarityUtil.getMentionConfidence(span2.getStr(), eLinkMap);
                        if (Math.abs(conf1 - conf2) < 0.0001) {//equal score
                            if (span1.getLength() < span2.getLength()) {//take longer
                                removeFlag[i] = true;
                            } else {
                                removeFlag[j] = true;
                            }
                        } else if (conf1 > conf2) {
                            if ((conf1 / conf2) < 1.25 && span2.covers(span1)) {//gap not very large, have another better mention
                                if (conf1 >= 0.99) { // span1 perfect match
                                    if (conf2 >= 0.9 && span2.getLength() >= 30) { //span2 very long and similarity very high
                                        removeFlag[i] = true; //delete span1
                                    } else {
                                        removeFlag[j] = true; //else delete span2
                                    }
                                } else {
                                    removeFlag[i] = true; // else delete span1
                                }
                            } else {
                                removeFlag[j] = true; //delete span that have smaller confidence
                            }
                        } else {
                            if ((conf2 / conf1) < 1.25 && span1.covers(span2)) {
                                if (conf2 >= 0.99) { // span2 perfect match
                                    if (conf1 >= 0.9 && span1.getLength() >= 30) { //span1 is very long and highly similar
                                        removeFlag[j] = true;
                                    } else {
                                        removeFlag[i] = true; //delete span1
                                    }
                                } else {
                                    removeFlag[j] = true; // else delete span2
                                }
                            } else {
                                removeFlag[i] = true; //delete span that have smaller confidence
                            }
                        }
                    }
                }
            }

            //Delete the mentions marked for deletion in eLinkMap
            for (int i = 0; i < spans.size(); i++) {
                if (removeFlag[i]) {
                    String key = spans.get(i).getStr();
                    eLinkMap.remove(key);
                }
            }
        }
        return eLinkMap;
    }

    /**
     * The entity link results identified by the re-scoring tool will advance the highest literal similarity and only return to top5
     *
     * @param entityLinkMap ELinkMap to be re-scored
     * @return eLinkMap
     */
    public static Map<String, List<Link>> reScoreEntity(Map<String, List<Link>> entityLinkMap) {

        for (Iterator<Map.Entry<String, List<Link>>> it = entityLinkMap.entrySet().iterator(); it.hasNext(); ) {

            Map.Entry<String, List<Link>> next = it.next();
            String key = next.getKey();
            List<Link> linkList = next.getValue();

            if (linkList == null || linkList.isEmpty()) {// Check if it is empty
                continue;
            } else {
                if (linkList.get(0).getType() == LinkEnum.RELATION) {// It is a relation, skip it. This function only reorders Entity
                    continue;
                }
            }

            // entity filtering
            if (Detector.isWhiteListFiltered())
                linkList.removeIf(link -> !Detector.getEntityWhiteList().contains(link.getUri()));

            // remove useless mention
            if (linkList.isEmpty()) {
                it.remove();
            }

            Iterator<Link> iter = linkList.iterator();
            while (iter.hasNext()) {
                Link link = iter.next();
                String mention = link.getMention();
                String uri = link.getUri();

                // filter by properties
                Set<String> properties = Detector.oneHopPropertyFiltered(uri);
                if (properties.isEmpty()) {//no property that in whitelist, delete
                    iter.remove();
                    continue;
                }

                // entity label
                String label = KBUtil.queryLabel(uri);
                if (label == null) {
                    label = UriUtil.splitLabelFromUri(uri);
                }

                // score before trimming the parentheses
                double lexScore1 = SimilarityUtil.getScore(mention, label);//Scoring without removing the parentheses of the disambiguation term

                // score after trimming the parentheses
                label = label.replaceAll("\\(.*", "").trim(); //Remove the brackets after the disambiguation item to avoid the disambiguation item is scored too low
                mention = mention.replaceAll("\\(.*", "").trim(); //Remove the disambiguation brackets in the mention
                double lexScore2 = SimilarityUtil.getScore(mention, label); //Score after removing parentheses

                // average lexical score
                double lexScore = (lexScore1 + lexScore2) / 2;

                // boost the score of entities contained by mention
                if (mention.toLowerCase().contains(label.toLowerCase().trim())) {//If mention contains label, increase the score appropriately
                    lexScore = Math.min(lexScore * 1.2, 1.0);
                }

                //For entities, it can be considered that literal similarity is the first choice
                //Set score to lexScore
                link.setScore(lexScore);
            }
            //Reorder to put the links with higher literal similarity on the front
            linkList.sort(Collections.reverseOrder());

            //Keep top6
            while (linkList.size() > 6) {
                linkList.remove(6);
            }

            //If the confidence is less than a certain threshold, remove the mention
            double mentionConfidence = SimilarityUtil.getMentionConfidence(key, entityLinkMap);
            if (mentionConfidence <= 0.5) {
                it.remove();
            }

        }

        return entityLinkMap;

    }

    /**
     * Re-scoring the relationship link Map to advance the similarity and keep top5
     *
     * @param rLinkMap RLinkMap to be re-scored
     * @return RlinkMap after re-scoring
     */
    public static Map<String, List<Link>> reScoreRelation(Map<String, List<Link>> rLinkMap) {
        //Simple scoring based on similarity
        for (String rMention : rLinkMap.keySet()) {
            if (!rLinkMap.get(rMention).isEmpty()) {
                List<Link> rLinkList = rLinkMap.get(rMention);
                if (rLinkList.get(0).getType() == LinkEnum.RELATION) { //Only deal with relation
                    Iterator<Link> linkIterator = rLinkList.iterator();
                    while (linkIterator.hasNext()) {
                        Link rLink = linkIterator.next();
                        String uri = rLink.getUri();
                        if (uri.startsWith("http://dbpedia.org/ontology/")) {
                            String[] strArr = uri.split("/");
                            String predicateName = strArr[strArr.length - 1];
                            if (Character.isUpperCase(predicateName.charAt(0))) {//Ontology with the first letter capitalized, such as dbo:Writer, usually Class, does not constitute a relationship
                                linkIterator.remove(); //Delete this link result
                            }
                        }

                        String label;
                        label = KBUtil.queryLabel(uri);
                        if (label == null) {
                            label = UriUtil.extractUri(uri);
                        }
                        //Based on dictionary and literal similarity calculation
                        double score = SimilarityUtil.getScoreWithParaphrase(label, rMention);
                        rLink.setScore(score);

                    }
                }
                reSortRelationLinkList(rLinkList); //Reorder, ontology advance
                //Keep top5
                while (rLinkList.size() > 5) {
                    rLinkList.remove(5);
                }
            }
        }
        return rLinkMap;
    }

    /**
     * Reorder the relational link list, put the same link dbo in advance and dbp in the back
     *
     * @param relationLinkList The relationLinkList to be sorted
     */
    public static void reSortRelationLinkList(List<Link> relationLinkList) {
        //First use fast sorting to sort from high to low
        relationLinkList.sort(Collections.reverseOrder());

        //Delete the ontology whose initials are capitalized, such as dbo:Writer, etc.
        Iterator<Link> iter = relationLinkList.iterator();
        while (iter.hasNext()) {
            Link link = iter.next();
            String uri = link.getUri();
            if (uri.startsWith("http://dbpedia.org/ontology/")) {
                String[] strArr = uri.split("/");
                String predicateName = strArr[strArr.length - 1];
                if (Character.isUpperCase(predicateName.charAt(0))) {//Ontology with the first letter capitalized, such as dbo:Writer, usually Class, does not constitute a relationship
                    iter.remove(); //Delete this link result
                }
            }
        }


        for (int i = 0; i < relationLinkList.size() - 1; i++) {
            Link link1 = relationLinkList.get(i);
            Link link2 = relationLinkList.get(i + 1);

            //The two links are the same, such as dbo:writer;dbp:writer, put dbo first
            if (UriUtil.splitLabelFromUri(link1.getUri()).equals(UriUtil.splitLabelFromUri(link2.getUri()))) {
                if (link1.getUri().startsWith("http://dbpedia.org/property/")
                        && link2.getUri().startsWith("http://dbpedia.org/ontology/")) {
                    // swap
                    relationLinkList.set(i, link2);
                    relationLinkList.set(i + 1, link1);
                }
            }
        }

    }

    /**
     * Check DBPedia if a label exists as the specified type
     *
     * @param label   the entity label
     * @param uriType the uri type, ontology or resource
     * @return true if exists, false otherwise
     */
    public static boolean uriExists(String label, String uriType) {
        String query = "ASK WHERE { <http://dbpedia.org/" + uriType + "/";
        query = query + label + "> ?p ?o } ";
        ArrayList<QuerySolution> querySolutions = KBUtil.runQuery(QueryFactory.create(query));
        if (querySolutions.isEmpty()) return false;
        return querySolutions.get(0).getLiteral("__ask_retval").toString().equals("1^^http://www.w3.org/2001/XMLSchema#integer");
    }

    public static void literallyDetection(String sentence, HashMap<String, List<Link>> resultMap) {
        if (uriExists(sentence, "ontology")) {
            resultMap.computeIfAbsent(sentence, k -> new ArrayList<>());
            resultMap.get(sentence).add(new Link(sentence, "http://dbpedia.org/ontology/" + sentence, LinkEnum.ENTITY, 1.0));
        }
        if (uriExists(sentence, "resource")) {
            resultMap.computeIfAbsent(sentence, k -> new ArrayList<>());
            resultMap.get(sentence).add(new Link(sentence, "http://dbpedia.org/resource/" + sentence, LinkEnum.ENTITY, 1.0));
        }
    }

    /**
     * Recognize long entities in questions, based on dexter, earl and falcon
     * At the same time identify the names of people, places, institutions, etc. in the question
     *
     * @param question  Question, and substitute the question
     * @param entityMap The mapping between the serial number after the substitution and the label, such as <e0>:Obama
     * @return Questions after long entity substitution
     */
    public static String recognizeLongEntity(String question, Map<String, String> entityMap, KBEnum kb) {


        try {

            //Results of identified entities
            int entityNum = 0;

            if (!entityMap.isEmpty()) {
                for (String key : entityMap.keySet()) {
                    String mention = entityMap.get(key);
                    if (mention.split(" ").length >= 3) {
                        question = question.replace(mention, key);
                    }
                    entityNum++;
                }
            }

            //Identify some noun phrases that are easily mis-segmented by the syntactic tree
            Matcher matcher1 = Pattern.compile("(?i)(television show(s)?)").matcher(question);
            if (matcher1.find()) {
                question = question.replace(matcher1.group(), "<e" + entityNum + ">");
                entityMap.put("<e" + entityNum + ">", matcher1.group());
                entityNum++;
            }

            // for DBpedia, use EARL for enrichment
            if (kb == KBEnum.DBpedia) {
                // use dexter for general question
                Map<String, List<Link>> dexterLinking = LinkingTool.getDexterLinking(question);
                //System.out.println("dexterLinking:" + dexterLinking);
                for (String mention : dexterLinking.keySet()) {
                    //System.out.println("earlMention:"+mention);
                    if (!dexterLinking.get(mention).isEmpty()) {
                        if (dexterLinking.get(mention).get(0).getType() == LinkEnum.ENTITY) { //Substitution entity only
                            int mentionLen = mention.split(" ").length;
                            if (mentionLen >= 3) {
                                if (!mention.trim().toLowerCase().startsWith("how many")
                                        && !mention.trim().toLowerCase().startsWith("give me")) { //how many nations will be misidentified
                                    //Identify disambiguation terms
                                    String regex = "(?i)" + mention.replace("+", "\\+").replace(".", "\\.") + "(\\s+\\(.*\\))";
                                    Matcher matcher = Pattern.compile(regex).matcher(question);
                                    if (matcher.find()) {
                                        mention = matcher.group();
                                    }
                                    entityMap.put("<e" + entityNum + ">", mention);

                                    //Replace with <e0> to prevent interference with the syntax tree
                                    question = question.replace(mention, "<e" + entityNum + ">");
                                    entityNum++;
                                }
                            }
                        }
                    }
                }

                // earl link result
                Map<String, List<Link>> earlLinking = getEARLLinking(question);
                //System.out.println("earlLinking:" + earlLinking);
                for (String mention : earlLinking.keySet()) {
                    //System.out.println("earlMention:"+mention);
                    if (!earlLinking.get(mention).isEmpty()) {
                        if (earlLinking.get(mention).get(0).getType() == LinkEnum.ENTITY) { //Substitution entity only
                            int mentionLen = mention.split(" ").length;
                            if (mentionLen >= 3) {

                                //Check if dexter has been recognized
                                if (mention.contains("< e")) {//dexter has been replaced and misidentified
                                    continue;
                                }

                                boolean dexterRecognized = false;
                                for (String dexterMention : dexterLinking.keySet()) {
                                    if (mention.contains(dexterMention)) {//dexter has been recognized, skip
                                        dexterRecognized = true;
                                        break;
                                    }
                                }
                                if (dexterRecognized) continue;

                                String regex = "(?i)" + mention.replace("+", "\\+").replace(".", "\\.") + "(\\s+\\(.*\\))";
                                Matcher matcher = Pattern.compile(regex).matcher(question);
                                if (matcher.find()) {
                                    mention = matcher.group();
                                }
                                entityMap.put("<e" + entityNum + ">", mention);

                                //Replace with <e0> to prevent syntax tree
                                question = question.replace(mention, "<e" + entityNum + ">");
                                entityNum++;
                            }
                        }
                    }
                }
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
        //System.out.println(question);
        return question;
    }

    /**
     * return entity and its candidateID detected by dexter2
     *
     * @param question natural language question
     * @return {mention:wikiID} map
     */
    public static HashMap<String, List<Integer>> getCandidateEntityIDs_dexter(String question) {

        HashMap<String, List<Integer>> result = new HashMap<>();

        String url;

        JSONObject inputObj = new JSONObject();
        inputObj.put("text", question);
        inputObj.put("wn", "false");
        inputObj.put("debug", "false");
        inputObj.put("format", "text");

        url = dexterLocalUrl;
        String jsonString = HttpsClientUtil.doPostWithParams(url, inputObj);

        if (jsonString != null && !jsonString.isEmpty()) {
            JSONObject jsonObject = new JSONObject(jsonString);
            //System.out.println(jsonObject.toString(4));
            JSONArray spots = jsonObject.getJSONArray("spots");
            // System.out.println(spots);
            for (int i = 0; i < spots.length(); i++) {
                List<Integer> res = new LinkedList<>();
                // the substring of question has the original case
                spots.getJSONObject(i).put("mention",
                        question.substring(spots.getJSONObject(i).getInt("start"), spots.getJSONObject(i).getInt("end")));
                String mention = spots.getJSONObject(i).getString("mention");
                double linkProbability = spots.getJSONObject(i).getDouble("linkProbability");
                if (linkProbability < 0.5) {
                    continue;
                }
                JSONArray candidates = spots.getJSONObject(i).getJSONArray("candidates");
                for (int j = 0; j < Math.min(candidates.length(), 5); j++) {//Only take top5
                    res.add(candidates.getJSONObject(j).getInt("entity"));
                }
                result.put(mention, res);
            }
        }
        return result;
    }

    /**
     * judge if a mention is dexter entity
     *
     * @param utterance mention
     * @param e         threshold of confidence
     * @return if confidence grater than threshold e, return true，else return false
     */
    public static boolean isDexterEntity(String utterance, double e) {

        JSONObject inputObj = new JSONObject();
        inputObj.put("text", utterance);
        inputObj.put("wn", "false");
        inputObj.put("debug", "false");
        inputObj.put("format", "text");

        String jsonString = HttpsClientUtil.doPostWithParams(dexterLocalUrl, inputObj);
        if (jsonString != null && !jsonString.isEmpty()) {
            JSONObject jsonObject = new JSONObject(jsonString);

            JSONArray spots = jsonObject.getJSONArray("spots");
            for (int i = 0; i < spots.length(); i++) {
                JSONObject spot = spots.getJSONObject(i);
                String mention = spot.getString("mention");
                //System.out.println(mention);
                if ((double) mention.length() / utterance.length() >= e) {
                    /*The proportion of the total length is greater than the threshold e*/
                    return true;
                }
            }
        }

        return false;
    }

    public static void main(String[] args) throws IOException {

        String question = "Where did these popular aeroplanes - Focke Wulf 260 and Start+Flug H 101 originate?";

        System.out.println("dexter linking:" + getDexterLinking(question));
        System.out.println("========================================");
        System.out.println("earl linking:" + getEARLLinking(question));
        System.out.println("========================================");
        System.out.println("falcon linking:" + getFalconLinking(question));
        System.out.println("========================================");


    }
}
