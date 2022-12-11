package cn.edu.nju.ws.edgqa.handler;

import cn.edu.nju.ws.edgqa.domain.beans.Link;
import cn.edu.nju.ws.edgqa.domain.beans.LinkMap;
import cn.edu.nju.ws.edgqa.domain.beans.tuple.ThreeTuple;
import cn.edu.nju.ws.edgqa.domain.beans.tuple.TwoTuple;
import cn.edu.nju.ws.edgqa.domain.edg.EDG;
import cn.edu.nju.ws.edgqa.domain.edg.Node;
import cn.edu.nju.ws.edgqa.domain.edg.SparqlGenerator;
import cn.edu.nju.ws.edgqa.eval.Evaluator;
import cn.edu.nju.ws.edgqa.main.QAArgs;
import cn.edu.nju.ws.edgqa.utils.NLPUtil;
import cn.edu.nju.ws.edgqa.utils.SimilarityUtil;
import cn.edu.nju.ws.edgqa.utils.UriUtil;
import cn.edu.nju.ws.edgqa.utils.enumerates.*;
import cn.edu.nju.ws.edgqa.utils.kbutil.KBUtil;
import cn.edu.nju.ws.edgqa.utils.linking.GoldenLinkingUtil;
import cn.edu.nju.ws.edgqa.utils.linking.LinkingTool;
import cn.edu.nju.ws.edgqa.utils.semanticmatching.NeuralSemanticMatchingScorer;
import com.google.common.collect.Ordering;
import org.checkerframework.checker.nullness.qual.Nullable;

import java.util.*;
import java.util.stream.Collectors;


/**
 * one question, one question solver
 * store intermediate results
 */
public class QuestionSolver {

    public static final double entityLinkThreshold = 0.3;
    public static final double relationLinkThreshold = 0.01;
    public final static Object lock = new Object();
    /**
     * The time limit for entity linking thread (seconds)
     */
    public static int entityLinkingThreadTimeLimit = 10;
    /**
     * The sub-sparql for each node
     */
    private final Map<Integer, List<SparqlGenerator>> subQuerySparqlMap;
    /**
     * The entity linking list for a node, top-1 mention
     * Simple Assumption: one node contains at most one entity and one corresponding relation
     * key: node id, value: entity linking list
     */
    private final Map<Integer, List<Link>> nodeEntityListMap;
    /**
     * The relation linking list for a node, top-1 mention
     * key: node id, value: relation linking list
     */
    private final Map<Integer, List<Link>> nodeRelationListMap;
    /**
     * The set of nodes which have no linked entities.
     */
    private final HashSet<Integer> nodeIdsWithNoEntity;
    /**
     * The type linking result of a node
     * key: node id, value: type linking list
     */
    private final Map<Integer, List<String>> nodeTypeListMap;
    /**
     * The generated candidate 3-tuple for each EDG node.
     * <p>
     * key: node id, value: (entity url, relation url, entity_score * relation_score)
     */
    private final HashMap<Integer, List<ThreeTuple<String, String, Double>>> nodeERTupleListMap;
    private final HashMap<Integer, List<TwoTuple<List<ThreeTuple<String, String, Double>>, Double>>> blockCandidateMap;
    private final LinkMap globalEntityLinkMap;
    private final LinkMap globalRelationLinkMap;
    /**
     * The natural language question
     */
    private String question;
    /**
     * The serial number in the dataset
     */
    private int serialNumber = -1;
    /**
     * The entity-description graph for this question
     */
    private EDG edg;
    /**
     * The results of sub-questions; key: node id, value: potential result
     */
    private Map<Integer, List<String>> subResultsMap;

    /**
     * Constructor
     *
     * @param question the natural language question
     * @param edg      the corresponding EDG instance
     */
    public QuestionSolver(String question, EDG edg) {
        this.question = question;
        this.edg = edg;
        this.subResultsMap = new HashMap<>();
        this.subQuerySparqlMap = new HashMap<>();
        this.nodeEntityListMap = new HashMap<>();
        this.nodeRelationListMap = new HashMap<>();
        this.nodeIdsWithNoEntity = new HashSet<>();
        this.nodeERTupleListMap = new HashMap<>();
        this.blockCandidateMap = new HashMap<>();
        this.nodeTypeListMap = new HashMap<>();
        this.globalEntityLinkMap = new LinkMap();
        this.globalRelationLinkMap = new LinkMap();
    }

    /**
     * Constructor
     *
     * @param question     the natural language question
     * @param serialNumber the question serial number in the dataset
     * @param edg          the corresponding EDG instance
     */
    public QuestionSolver(String question, int serialNumber, EDG edg) {
        this.question = question;
        this.serialNumber = serialNumber;
        this.edg = edg;
        this.subResultsMap = new HashMap<>();
        this.subQuerySparqlMap = new HashMap<>();
        this.nodeEntityListMap = new HashMap<>();
        this.nodeRelationListMap = new HashMap<>();
        this.nodeIdsWithNoEntity = new HashSet<>();
        this.nodeERTupleListMap = new HashMap<>();
        this.blockCandidateMap = new HashMap<>();
        this.nodeTypeListMap = new HashMap<>();
        this.globalEntityLinkMap = new LinkMap();
        this.globalRelationLinkMap = new LinkMap();
    }

    public static int getEntityLinkingThreadTimeLimit() {
        return entityLinkingThreadTimeLimit;
    }

    public static void setEntityLinkingThreadTimeLimit(int entityLinkingThreadTimeLimit) {
        QuestionSolver.entityLinkingThreadTimeLimit = entityLinkingThreadTimeLimit;
    }

    public String getQuestion() {
        return question;
    }

    public void setQuestion(String question) {
        this.question = question;
    }

    public EDG getEdg() {
        return edg;
    }

    public void setEdg(EDG edg) {
        this.edg = edg;
    }

    public Map<Integer, List<String>> getSubResultsMap() {
        return subResultsMap;
    }

    public void setSubResultsMap(HashMap<Integer, List<String>> subResultsMap) {
        this.subResultsMap = subResultsMap;
    }

    public Map<Integer, List<SparqlGenerator>> getSubQuerySparqlMap() {
        return subQuerySparqlMap;
    }

    /**
     * The question solving process
     *
     * @return the predicted answer list
     */
    public List<String> solveQuestion() {


        List<String> result = new ArrayList<>(); // predicted answer

        System.out.println("Fetching global linking...");
        LinkingTool.getEnsembleLinking(edg.getQuestion(), globalEntityLinkMap.getData(), globalRelationLinkMap.getData(), false); // global linking results
        System.out.println("Global Linking Fetched");

        if (this.edg.getNumEntity() <= 0) { // EDG generation failed
            System.out.println("[ERROR] EDG generation failed: " + edg.getQuestion());
            return result;
        } else { // start from the root entity

            List<SparqlGenerator> sparqlList;
            // specially for judgement
            if (edg.getQueryType() == QueryType.JUDGE) {

                sparqlList = judgeQuestionSolver(edg, 0);
                EvaluateLinking(sparqlList);
                for (SparqlGenerator spg : sparqlList) {
                    result.addAll(spg.solve());
                }

                // distinct result
                if (result.contains("true")) {
                    result.clear();
                    result.add("true");
                } else {
                    result.clear();
                    result.add("false");
                }

            } else { // not boolean question, all other types of questions

                if (QAArgs.isQuestionDecomposition()) {  // with question decomposition
                    sparqlList = compoundQuestionSolver(edg, 0);
                } else {  // without question decomposition
                    edg = edg.flattenEDG();
                    sparqlList = atomicQuestionSolver(edg, 0);
                    subQuerySparqlMap.put(0, sparqlList);
                }
                for (SparqlGenerator spg : sparqlList) {
                    result.addAll(spg.solve());
                }
            }
        }

        return result.stream().distinct().collect(Collectors.toList());
    }

    /**
     * Deal with all entity blocks with referred entities
     *
     * @param edg      edg
     * @param entityID index of the entity block
     * @return all possible sparql queries
     */
    public List<SparqlGenerator> compoundQuestionSolver(EDG edg, int entityID) {

        List<SparqlGenerator> sparqlList = new ArrayList<>();  // generated sparql list

        List<Node> relatedDescription = edg.getRelatedDescription(entityID);//Get all connected Description Node 
        HashSet<Integer> relatedDescriptionIDs = new HashSet<>();//Node IDS related to current node 
        relatedDescription.forEach(node -> relatedDescriptionIDs.add(node.getNodeID()));

        if (!edg.entityNodeHasRefer(entityID)) { // no referred entity, call atomic question solver
            List<SparqlGenerator> subSparqlList = atomicQuestionSolver(edg, entityID);
            sparqlList.addAll(subSparqlList);
            subQuerySparqlMap.put(entityID, subSparqlList);

            // remove empty query
            sparqlList.removeIf(sparqlGenerator -> sparqlGenerator.getTupleList().isEmpty());

            if (entityID == 0) { // root entity, evaluate linking
                EvaluateLinking(sparqlList);
            }

            return sparqlList;
        } else { // if it has referred entity, deal with a composed question

            //all entityID referred from this entity
            List<Integer> referredEntity = edg.getReferredEntity(entityID);

            // Iterate all sub entities
            for (Integer subEntityID : referredEntity) {
                List<SparqlGenerator> subSparqlList = compoundQuestionSolver(edg, subEntityID); // recursively solve this sub entity
                subQuerySparqlMap.put(subEntityID, subSparqlList);
                List<String> subResult = new ArrayList<>();
                for (SparqlGenerator subSparqlGen : subSparqlList) {
                    List<String> subResultList = subSparqlGen.solve();
                    for (String res : subResultList) {
                        if (QAArgs.getDataset() != DatasetEnum.LC_QUAD || res.startsWith("http://dbpedia.org/resource")) {
                            subResult.add(res);
                        }
                    }
                }
                subResult = subResult.stream().distinct().collect(Collectors.toList()); // remove the duplicate

                if (subEntityID > 0) { //not root
                    while (subResult.size() > 20) {// limit the number of subResult to speed up generation
                        subResult.remove(20);
                    }
                }

                subResultsMap.put(subEntityID, subResult);
            }

            System.out.println("SubQuerySparql:" + subQuerySparqlMap);
            System.out.println("SubQueryResult:" + subResultsMap);

            // start replacement
            // detect entities of all nodes without reference edge
            for (Node desNode : relatedDescription) {
                if (!desNode.isContainsRefer()) {
                    initialEntityDetection(desNode);
                }
            }

            // reduce irrelevant entities
            entityReduce(relatedDescriptionIDs);

            // relation detection of all nodes without reference edge
            for (Node desNode : relatedDescription) {
                if (!desNode.isContainsRefer()) {
                    initialRelationDetection(desNode);
                }
            }

            //Node generated candidate TUPLE for the result of the solid link 
            for (Node desNode : relatedDescription) {
                if (!desNode.isContainsRefer()) {
                    List<ThreeTuple<String, String, Double>> tupleList = nodeTupleGeneration(desNode, relatedDescriptionIDs);
                    if (!tupleList.isEmpty()) nodeERTupleListMap.put(desNode.getNodeID(), tupleList);
                }
            }

            //type detection of nodes without reference edge
            if (QAArgs.isUsingTypeConstraints()) {
                for (Node desNode : relatedDescription) {
                    if (!desNode.isContainsRefer()) {
                        if (nodeIdsWithNoEntity.contains(desNode.getNodeID())) {
                            List<String> typeList = Detector.typeDetection(desNode.getStr());
                            if (!typeList.isEmpty()) {
                                nodeTypeListMap.put(desNode.getNodeID(), typeList);
                            }
                        }
                    }
                }
            }

            //handle description nodes with reference edge
            for (Node desNode : relatedDescription) {//Generate Candidate Tuple by node 
                int nodeID = desNode.getNodeID();
                if (desNode.isContainsRefer()) {//The current node contains REFER 
                    int desReferredEntityID = edg.getDesReferredEntityID(desNode.getNodeID());
                    List<SparqlGenerator> subSparqlList = subQuerySparqlMap.get(desReferredEntityID);
                    String nodeStr = desNode.getStr();
                    if (subSparqlList.isEmpty()) {
                        continue;
                    }
                    if (SimilarityUtil.isDescriptionEqual(nodeStr, "#entity" + desReferredEntityID)) {
                        //This is redundant decomposition, directly return answers 
                        subSparqlList.forEach(sp -> {
                            SparqlGenerator newSP = new SparqlGenerator(sp);
                            newSP.changeFinalVarName("e" + entityID);
                            sparqlList.add(newSP);
                        });
                    } else {//Non-redundant decomposition 
                        List<String> subResults = subResultsMap.get(desReferredEntityID);
                        if (!subResults.isEmpty()) {
                            // Subproof results 
                            // Replace the #ntity {i} in Nodestr and join NodelistMap 
                            String entity_id_mention = "#entity" + desReferredEntityID;
                            List<Link> linkList = new ArrayList<>();
                            subResults.forEach(entityUri -> linkList.add(new Link(entity_id_mention, entityUri, LinkEnum.ENTITY, 1.0)));

                            nodeEntityListMap.put(nodeID, linkList);

                            Map<String, List<Link>> entityMap = new HashMap<>();
                            entityMap.put(entity_id_mention, linkList);
                            //nodeEntityMap.put(nodeID, entityMap);
                            desNode.setStr(nodeStr.replaceAll(entity_id_mention, "<e0>").trim());

                            Map<String, String> entityIndexMap = new HashMap<>();
                            entityIndexMap.put("<e0>", entity_id_mention);

                            // relation Detection
                            Trigger trigger = Trigger.UNKNOWN;
                            if (edg.findEntityBlockID(desNode) == 0) {
                                trigger = edg.getTrigger();
                            }
                            Map<String, List<Link>> relationMap = Detector.relationDetection(desNode.getStr(),
                                    entityIndexMap, nodeEntityListMap.get(nodeID), trigger,
                                    QAArgs.getGoldenLinkingMode(),
                                    GoldenLinkingUtil.getGoldenRelationLinkBySerialNumber(serialNumber), this.globalRelationLinkMap);// Here is Refer Node's Relation Detection 

                            if (!relationMap.isEmpty()) { //relationMap is not empty

                                //nodeRelationMap.put(nodeID, relationMap);
                                for (String rMention : relationMap.keySet()) {
                                    nodeRelationListMap.put(nodeID, relationMap.get(rMention));
                                    break;
                                }

                                List<Link> rlinkList = nodeRelationListMap.getOrDefault(nodeID, null);

                                List<String> rlinkUriList = new ArrayList<>();
                                rlinkList.forEach(o -> rlinkUriList.add(o.getUri()));

                                HashSet<String> oneHopProperty = new HashSet<>();
                                for (Link eLink : linkList) {
                                    oneHopProperty.addAll(Detector.oneHopPropertyFiltered(eLink.getUri()));
                                }
                                //high similarity
                                Map<String, List<String>> labelUriMap = UriUtil.extractLabelMap(new ArrayList<>(oneHopProperty));


                                for (Integer id : nodeIdsWithNoEntity) {//Other nodes of the unintegrated link 
                                    if (relatedDescriptionIDs.contains(id)) {
                                        List<Link> otherRLink = nodeRelationListMap.get(id);
                                        if (otherRLink != null) {
                                            for (Link otherR : otherRLink) {
                                                if (rlinkUriList.contains(otherR.getUri())) {//Re-score, improve 
                                                    //rlinkList.get(rlinkList.indexOf(otherR)).setScore(0.6);
                                                    rlinkList.removeIf(o -> o.getUri().equals(otherR.getUri()));
                                                    otherR.setScore(0.6);
                                                    rlinkList.add(otherR);
                                                } else if (oneHopProperty.contains(otherR.getUri())) {
                                                    rlinkList.add(new Link(otherR.getMention(), otherR.getUri(), LinkEnum.RELATION, 0.8));
                                                }
                                            }
                                        }


                                        String otherStr = edg.getNodes()[id].getStr();
                                        HashMap<String, Double> labelSim = SimilarityUtil.getCompositeSetSimilarity(otherStr, labelUriMap.keySet());
                                        for (String propLabel : labelSim.keySet()) {
                                            double score = labelSim.get(propLabel);
                                            if (score > 0.6) {
                                                labelUriMap.get(propLabel).forEach(property ->
                                                        rlinkList.add(new Link(otherStr, property, LinkEnum.RELATION, score)));

                                            }
                                        }

                                    }
                                }
                                rlinkList.sort(Collections.reverseOrder()); //Reorder, from high to low 

                                for (Link rLink : rlinkList) {
                                    for (SparqlGenerator subSparql : subSparqlList) {//Each result of the child is added to a Triple 
                                        SparqlGenerator newSparql = new SparqlGenerator(subSparql);
                                        String oldVariable = newSparql.getFinalVarName();
                                        String newVariable = "e" + entityID;
                                        newSparql.setFinalVarName(newVariable);
                                        // Add one? E0 <dbp: xxx>? E1 
                                        newSparql.addTriple("?" + newVariable, "<" + rLink.getUri() + ">", "?" + oldVariable);
                                        newSparql.setScore(newSparql.getScore() * rLink.getScore());
                                        sparqlList.add(newSparql);
                                    }
                                }
                            } else { // if relationMap is Empty
                                subSparqlList.forEach(subSparql -> sparqlList.add(new SparqlGenerator(subSparql)));
                            }
                        }
                    }
                }
            }

            // root block, modify the query type of sparql
            if (entityID == 0) {
                if (sparqlList.isEmpty()) {//empty sparqlList, add new sparql
                    SparqlGenerator spg = new SparqlGenerator();
                    spg.setFinalVarName("e0");
                    sparqlList.add(spg);
                }
                if (edg.getNodes()[0].getQueryType() == QueryType.JUDGE) {
                    sparqlList.forEach(sparqlGenerator -> sparqlGenerator.setQuesType(QueryType.JUDGE));
                } else if (edg.getNodes()[0].getQueryType() == QueryType.COUNT) {
                    sparqlList.forEach(sparqlGenerator -> sparqlGenerator.setQuesType(QueryType.COUNT));
                } else {
                    sparqlList.forEach(sparqlGenerator -> sparqlGenerator.setQuesType(QueryType.COMMON));
                }
            }

            List<SparqlGenerator> newSparqlList = new ArrayList<>();

            //Add other candidate Tuples of the current node into Sparql 
            HashSet<Integer> tempSet = new HashSet<>(nodeERTupleListMap.keySet());
            tempSet.retainAll(relatedDescriptionIDs);
            if (!tempSet.isEmpty()) {
                for (Integer nodeID : tempSet) {
                    List<ThreeTuple<String, String, Double>> nodeTuples = nodeERTupleListMap.get(nodeID);
                    for (SparqlGenerator sparql : sparqlList) {
                        for (ThreeTuple<String, String, Double> nodeTuple : nodeTuples) {
                            SparqlGenerator newSparql = new SparqlGenerator(sparql);
                            newSparql.addTriple("?e" + entityID, "<" + nodeTuple.getSecond() + ">", "<" + nodeTuple.getFirst() + ">");
                            newSparql.setScore(newSparql.getScore() * nodeTuple.getThird()); // reset the score
                            if (!newSparql.execute().isEmpty()) {
                                newSparqlList.add(newSparql);
                            }
                        }
                    }
                }
            }

            if (!newSparqlList.isEmpty()) {
                sparqlList.clear();
                sparqlList.addAll(newSparqlList);
            }
            //sparqlList.removeIf(sp->KBUtil.isZeroSelect(sp.toString()));

            sparqlListDistinct(sparqlList);

            // potential type constraint
            if (QAArgs.isUsingTypeConstraints()) {
                addTypeConstraint(relatedDescriptionIDs, sparqlList);
            }

            // reRank the sparqls
            if (QAArgs.isReRanking()) {
                reRankSparql(entityID, sparqlList);
            }

            // reduce the number of candidate sparqls
            reduceCandidateSparql(entityID, sparqlList);


            //expand query
            if (QAArgs.getGoldenLinkingMode() == GoldenMode.DISABLED && !QAArgs.isEvaluateCandNum()) {
                List<SparqlGenerator> temp = new ArrayList<>(sparqlList);
                sparqlList.clear();
                temp.forEach(sp -> sparqlList.addAll(sp.expandQueryWithDbpOrDbo()));
            }

            List<SparqlGenerator> distinctList = sparqlList.stream().distinct().collect(Collectors.toList());
            sparqlList.clear();
            sparqlList.addAll(distinctList);

            sparqlList.removeIf(sparqlGenerator -> sparqlGenerator.getTupleList().isEmpty());

            //Add to SubQuerySparqlmap 
            subQuerySparqlMap.put(entityID, sparqlList);

            if (entityID == 0) {
                /* final */
                EvaluateLinking(sparqlList);
            }
            return sparqlList;
        } // end if it has referred entities

    }

    /**
     * remove redundant sparql candidates in the sparql List, retain the higher score
     *
     * @param sparqlList sparqlList
     */
    private void sparqlListDistinct(List<SparqlGenerator> sparqlList) {
        //distinct the sparql List
        ListIterator<SparqlGenerator> iter = sparqlList.listIterator(sparqlList.size());
        while (iter.hasPrevious()) {
            SparqlGenerator previous = iter.previous();
            int index = iter.previousIndex();
            for (int i = index; i >= 0; i--) {
                if (sparqlList.get(i).equals(previous)) {
                    iter.remove();
                    break;
                }
            }
        }
    }

    /**
     * solving an atomic question block, return possible sparqls
     *
     * @param edg      edg
     * @param entityID block id
     * @return sparqls for
     */
    public List<SparqlGenerator> atomicQuestionSolver(EDG edg, int entityID) {
        System.out.println("[DEBUG] (AQS) Processing EntityID: " + entityID);

        //Description nodes in current block
        List<Node> relatedDes = edg.getRelatedDescription(entityID);
        Set<Integer> relatedDesIDs = relatedDes.stream().map(Node::getNodeID).collect(Collectors.toSet());

        //Entity Detection Part
        for (Node node : relatedDes) {//Entity detection of the node one by one 
            initialEntityDetection(node); //shallowEntityDetection£¬Determine nodeemap 
        }
        //System.out.println("NodeEListMap after initial detection:" + nodeEntityListMap);

        if (!QAArgs.isGoldenEntity()) {
            entityReduce(relatedDesIDs); //reduce the irrelevant entities
        }
        //System.out.println("NodeEListMap after entity reduce:" + nodeEntityListMap);

        //Relation Detection Part
        for (Node node : relatedDes) {//Starting a node Relation detection 
            initialRelationDetection(node);
        }

        System.out.println("nodeEListMap:" + nodeEntityListMap);
        System.out.println("nodeRListMap:" + nodeRelationListMap);

        // Type Detection Part
        if (QAArgs.isUsingTypeConstraints()) {
            for (Node node : relatedDes) { // descriptive nodes in this block
                if (nodeIdsWithNoEntity.contains(node.getNodeID())) { // no entity, just relation
                    List<String> typeList = Detector.typeDetection(node.getStr());
                    if (!typeList.isEmpty()) {
                        nodeTypeListMap.put(node.getNodeID(), typeList);
                    }
                }
            }
            System.out.println("nodeTypeListMap:" + nodeTypeListMap);
        }

        // Assemble entities and relations to build queries
        for (Node node : relatedDes) {
            List<ThreeTuple<String, String, Double>> tupleList = nodeTupleGeneration(node, relatedDesIDs);
            if (!tupleList.isEmpty()) nodeERTupleListMap.put(node.getNodeID(), tupleList);
        }

        System.out.println("nodeERTupleList:" + nodeERTupleListMap);

        //Generate Candidate Tuple of Block 
        blockCandidateMap.put(entityID, blockTupleGeneration(relatedDesIDs));

        //generate sparql for this atomic block, sorted DESC
        List<SparqlGenerator> sparqlList = blockSparqlGeneration(entityID, edg, relatedDesIDs);
        sparqlList.sort(Collections.reverseOrder());

        //distinct the sparqlList
        sparqlListDistinct(sparqlList);

        // potential type constraint
        if (QAArgs.isUsingTypeConstraints()) {
            addTypeConstraint(relatedDesIDs, sparqlList);
        }

        // reRanking sparqls with block
        if (QAArgs.isReRanking()) {
            reRankSparql(entityID, sparqlList);
        }

        // for QALD, the system needs to judge whether the question can be answered
        if (QAArgs.getDataset() == DatasetEnum.QALD_9) {
            double total_threshold = 0.25;
            double rel_threshold = 0.15;
            double threshold;

            boolean isEntityDetected = false;
            List<Link> entityLinking = nodeEntityListMap.values().stream().flatMap(List::stream).collect(Collectors.toList());
            for (Link link : entityLinking) {
                if (link.getScore() == 1.0) {
                    isEntityDetected = true;
                    break;
                }
            }
            if (isEntityDetected) {
                threshold = rel_threshold;
            } else {
                threshold = total_threshold;
            }


            if (!sparqlList.isEmpty()) {
                double maxScore = sparqlList.get(0).getScore();
                if (maxScore < threshold) {
                    sparqlList.clear();
                }
            }
        }

        //reduce the number of Candidate Sparqls
        reduceCandidateSparql(entityID, sparqlList);

        // dbo/dbp expand
        if (QAArgs.getGoldenLinkingMode() == GoldenMode.DISABLED && !QAArgs.isEvaluateCandNum()) {
            List<SparqlGenerator> expand = new ArrayList<>(sparqlList);
            sparqlList.clear();
            List<SparqlGenerator> tempList = new ArrayList<>();
            expand.forEach(sp -> tempList.addAll(sp.expandQueryWithDbpOrDbo()));
            sparqlList = tempList.stream().distinct().collect(Collectors.toList());
        }

        // sparql generation failed, generate default sparql
        if (sparqlList.isEmpty()) {

            if (QAArgs.getDataset() == DatasetEnum.QALD_9) {

                // for qald, try to find the concept constraint
                if (!nodeTypeListMap.isEmpty()) {
                    List<String> typeList = nodeTypeListMap.values().stream().flatMap(List::stream).distinct().collect(Collectors.toList());
                    SparqlGenerator spg = new SparqlGenerator();
                    spg.setQuesType(QueryType.COMMON);
                    String finalVarName = "e" + entityID;
                    spg.setFinalVarName(finalVarName);
                    spg.addTriple("?" + finalVarName, "http://www.w3.org/1999/02/22-rdf-syntax-ns#type", typeList.get(0));
                    sparqlList.add(spg);
                }

            } else {

                List<Link> candidateEntities = new ArrayList<>();
                for (List<Link> links : globalEntityLinkMap.getData().values()) {
                    candidateEntities.addAll(links);
                }

                Set<String> candidateRelations = new HashSet<>();
                globalRelationLinkMap.getData().values().forEach(list -> candidateRelations.addAll(list.stream().map(Link::getUri).collect(Collectors.toList())));

                for (Link entity : candidateEntities) {
                    Set<String> oneHops = KBUtil.oneHopProperty(entity.getUri());
                    oneHops.retainAll(candidateRelations);
                    if (!oneHops.isEmpty()) {
                        for (String oneHop : oneHops) {
                            SparqlGenerator spg = new SparqlGenerator();
                            if (entityID == 0) {
                                spg.setQuesType(edg.getQueryType());
                            } else {
                                spg.setQuesType(QueryType.COMMON);
                            }
                            String finalVarName = "e" + entityID;
                            spg.setFinalVarName(finalVarName);
                            spg.addTriple("?" + finalVarName, entity.getUri(), oneHop);
                            sparqlList.add(spg);
                        }
                    }
                }
                sparqlList = sparqlList.stream().distinct().collect(Collectors.toList());
                while (sparqlList.size() > 5) {
                    sparqlList.remove(5);
                }
            }

        }

        return sparqlList;
    }

    /**
     * add potential type constraint for candidate sparqls
     *
     * @param relatedDesIDs node ids in current block
     * @param sparqlList    candidate sparqls
     */
    private void addTypeConstraint(Set<Integer> relatedDesIDs, List<SparqlGenerator> sparqlList) {

        // get all the potential types
        List<String> typeList = new ArrayList<>();
        for (Integer typeNodeID : nodeTypeListMap.keySet()) {
            if (relatedDesIDs.contains(typeNodeID)) {
                typeList.addAll(nodeTypeListMap.get(typeNodeID));
            }
        }

        ListIterator<SparqlGenerator> iter = sparqlList.listIterator();
        List<SparqlGenerator> newSparqlList = new ArrayList<>();
        while (iter.hasNext()) {
            SparqlGenerator sparql = iter.next();
            //whether a type constraint is added
            for (String type : typeList) {
                SparqlGenerator sp = new SparqlGenerator(sparql);
                sp.addTriple("?" + sp.getFinalVarName(), "http://www.w3.org/1999/02/22-rdf-syntax-ns#type", "<" + type + ">");
                if (!KBUtil.isZeroSelect(sp.toString())) {
                    //query not empty
                    //sparqlList.set(iter.previousIndex(), sp);
                    newSparqlList.add(sp);
                    break;
                }
            }
            /*if(!added){
                newSparqlList.add(sparql);
            }*/
            //if (added) break;
        }

        // type added to current sparql, update it
        if (!newSparqlList.isEmpty()) {
            sparqlList.clear();
            sparqlList.addAll(newSparqlList);
        }

    }

    /**
     * reRank the candidate sparqls by block and sparql matching
     *
     * @param entityID   entityID
     * @param sparqlList candidate sparqlList
     */
    private void reRankSparql(int entityID, List<SparqlGenerator> sparqlList) {
        if (sparqlList.isEmpty()) return;
        HashMap<String, List<SparqlGenerator>> stringMap = new HashMap<>();
        for (SparqlGenerator spg : sparqlList) {

            String sparqlString;
            if (QAArgs.getDataset() == DatasetEnum.QALD_9) {
                sparqlString = spg.tupleListToString_QALD();
            } else {
                sparqlString = spg.tupleListToString_LCQUAD();
            }
            if (!stringMap.containsKey(sparqlString)) {
                stringMap.put(sparqlString, new ArrayList<>());
            }
            stringMap.get(sparqlString).add(spg);
        }
        String blockString = edg.blockToString(entityID);

        Map<String, Double> sparqlRerankScoreMap = NeuralSemanticMatchingScorer.query_reranking_score(blockString, stringMap.keySet());
        System.out.println("Block String:" + blockString);
        System.out.println("Sparql Rerank Map:" + sparqlRerankScoreMap);

        sparqlList.clear();
        for (String sparqlString : sparqlRerankScoreMap.keySet()) {
            List<SparqlGenerator> sparqlGeneratorList = stringMap.get(sparqlString);
            for (SparqlGenerator spg : sparqlGeneratorList) {
                spg.setScore((spg.getScore() + sparqlRerankScoreMap.get(sparqlString)) / 2); // avg score
                sparqlList.add(spg);
            }
        }
        sparqlList.sort(Collections.reverseOrder());

    }

    /**
     * reduce the number of candidate sparql
     *
     * @param entityID   entityID
     * @param sparqlList candidate sparql list
     */
    private void reduceCandidateSparql(int entityID, List<SparqlGenerator> sparqlList) {
        int upperBound = QAArgs.getCandidateSparqlNumUpperB();
        // only retain one for root
        if (entityID == 0) upperBound = QAArgs.getRootCandidateSparqlNumUpperB();

        double threshold;

        if (sparqlList.isEmpty()) {
            return;
        } else if (sparqlList.size() >= upperBound) {
            threshold = sparqlList.get(upperBound - 1).getScore();
        } else {
            threshold = sparqlList.get(0).getScore();
        }


        while (sparqlList.size() > upperBound) {
            SparqlGenerator tail = sparqlList.get(sparqlList.size() - 1);
            if (tail.getScore() >= threshold) {
                break;
            } else {
                sparqlList.remove(sparqlList.size() - 1);
            }
        }

        // if the first sparql is highly possible, only retain this one
        if (sparqlList.size() > 1 && !QAArgs.isEvaluateCandNum()) {
            SparqlGenerator firstSparql = sparqlList.get(0);
            Double firstScore = firstSparql.getScore();
            SparqlGenerator secondSparql = sparqlList.get(1);
            Double secondScore = secondSparql.getScore();
            if (firstScore > 0.9 && (secondScore < firstScore * 0.4)) {
                sparqlList.remove(1);
            }
        }

    }

    /**
     * solver for judgement questions specially
     *
     * @param edg      edg
     * @param entityID entityID, should be 0
     * @return all the possible for this question
     */
    public List<SparqlGenerator> judgeQuestionSolver(EDG edg, int entityID) {

        List<SparqlGenerator> sparqlList = new ArrayList<>();
        // all the entity candidates
        List<List<Link>> eLinkCands = null;
        //all the relations
        List<Link> rLinkList = null;

        if (!edg.entityNodeHasRefer(entityID)) {// simple edg block

            List<Node> relatedDesNodes = edg.getRelatedDescription(entityID);
            Set<Integer> relatedDesIDs = relatedDesNodes.stream().map(Node::getNodeID).collect(Collectors.toSet());

            for (Node node : relatedDesNodes) {
                initialEntityDetection(node);
            }
            if (!QAArgs.isGoldenEntity()) {
                entityReduce(relatedDesIDs);
            }
            for (Node node : relatedDesNodes) {
                initialRelationDetection(node);
            }

            eLinkCands = new ArrayList<>(nodeEntityListMap.values());
            rLinkList = nodeRelationListMap.values().stream().flatMap(Collection::stream).collect(Collectors.toList());
        } else {// compound edg block, use global linking
            eLinkCands = new ArrayList<>(globalEntityLinkMap.getData().values());
            rLinkList = globalRelationLinkMap.getData().values().stream().flatMap(Collection::stream).collect(Collectors.toList());
        }

        //default judge sparql
        SparqlGenerator spg = new SparqlGenerator();
        spg.setQuesType(QueryType.JUDGE);
        String varName = "e" + entityID;
        spg.setFinalVarName(varName);


        int eNodeNum = eLinkCands.size(); // Identified Entity Mention 
        if (eNodeNum < 1) {//No Entity 
            System.out.println("[ERROR] No entity detected for Question:" + question);
        } else if (eNodeNum == 1) {//Only one node recognizes Entity 
            List<Link> eLinkList = eLinkCands.get(0);
            for (Link eLink : eLinkList) {
                boolean bingoTag = false;
                for (Link rLink : rLinkList) {
                    if (KBUtil.isERexists(eLink.getUri(), rLink.getUri())) {
                        spg.addTriple("?" + varName, "<" + rLink.getUri() + ">", "<" + eLink.getUri() + ">");
                        spg.setFinalVarName(varName);
                        spg.setScore(eLink.getScore() * rLink.getScore());
                        bingoTag = true;
                        break;
                    }
                }
                if (bingoTag) break;
            }
            if (!spg.getTupleList().isEmpty()) {
                sparqlList.add(spg);
            }

        } else {//More than two Node recognizes Entity 
            List<Link> eLink1List = eLinkCands.get(0);
            List<Link> eLink2List = eLinkCands.get(1);
            boolean bingoTag = false;
            for (Link eLink1 : eLink1List) {
                for (Link eLink2 : eLink2List) {
                    for (Link rLink : rLinkList) {

                        // relations in rLinkList, return true
                        if (KBUtil.isE1RE2exists(eLink1.getUri(), rLink.getUri(), eLink2.getUri())) {
                            spg.addTriple("<" + eLink1.getUri() + ">", "<" + rLink.getUri() + ">", "<" + eLink2.getUri() + ">");
                            spg.setScore(eLink1.getScore() * eLink2.getScore() * rLink.getScore());
                            bingoTag = true;
                            break;
                        }

                        // relations not in rLinkList, find potential relation
                        if (KBUtil.isEPairInOneHop(eLink1.getUri(), eLink2.getUri())) {
                            List<String> rUris = KBUtil.getRelBetweenEPair(eLink1.getUri(), eLink2.getUri());

                            if (rUris.size() == 0) {
                                // all the relations between E1&E2 are wikiLinks
                                bingoTag = true;
                                spg.addTriple("<" + eLink1.getUri() + ">", "?p", "<" + eLink2.getUri() + ">");
                            } else {
                                //evaluate the similarity between the surface and the relation
                                Map<String, List<String>> labelMap = UriUtil.extractLabelMap(rUris);
                                String surface = question.replaceAll("(?i)" + eLink1.getMention(), "")
                                        .replaceAll("(?i)" + eLink2.getMention(), "")
                                        .replaceAll(" +", " ").trim();
                                HashMap<String, Double> similarity = SimilarityUtil.getCompositeSetSimilarity(surface, labelMap.keySet());
                                for (String label : similarity.keySet()) {
                                    if (similarity.get(label) > 0.3) {// the similarity is beyond a threshold
                                        spg.addTriple("<" + eLink1.getUri() + ">", "<" + labelMap.get(label).get(0) + ">", "<" + eLink2.getUri() + ">");
                                        spg.setScore(eLink1.getScore() * eLink2.getScore() * similarity.get(label));
                                        bingoTag = true;
                                        break;
                                    }
                                }
                            }
                        }
                        if (bingoTag) break;
                    }
                    if (bingoTag) break;
                }
                if (bingoTag) break;
            }
            if (!spg.getTupleList().isEmpty()) {
                sparqlList.add(spg);
            }
        }

        subQuerySparqlMap.put(entityID, sparqlList);
        return sparqlList;
    }

    /**
     * evaluate the performance of entity/relation/type linking
     *
     * @param sparqlList candidate sparql
     */
    public void EvaluateLinking(List<SparqlGenerator> sparqlList) {
        // entity/relation/type linking metrics
        List<String> entityPrediction = new ArrayList<>();
        List<String> relationPrediction = new ArrayList<>();
        List<String> typePrediction = new ArrayList<>();
        List<String> entityGolden = GoldenLinkingUtil.getGoldenEntityLinkURIBySerialNumber(serialNumber);
        List<String> relationGolden = GoldenLinkingUtil.getGoldenRelationLinkURIBySerialNumber(serialNumber);
        List<String> typeGolden = GoldenLinkingUtil.getGoldenTypeLinkURIBySerialNumber(serialNumber);
        for (SparqlGenerator sparqlGenerator : sparqlList) {
            for (ThreeTuple<String, String, String> triple : sparqlGenerator.getTupleList()) {
                if (triple.getFirst().startsWith("<http")) {
                    if (!triple.getFirst().contains("ontology")) entityPrediction.add(triple.getFirst());
                    if (triple.getSecond().contains("rdf-syntax-ns#type") && triple.getFirst().contains("ontology") && Character.isUpperCase(triple.getFirst().charAt(29)))
                        typePrediction.add(triple.getFirst());
                }
                if (triple.getThird().startsWith("<http")) {
                    if (!triple.getThird().contains("ontology")) entityPrediction.add(triple.getThird());
                    if (triple.getSecond().contains("rdf-syntax-ns#type") && triple.getThird().contains("ontology") && Character.isUpperCase(triple.getThird().charAt(29)))
                        typePrediction.add(triple.getFirst());
                }
                if (triple.getSecond().startsWith("<http") && !triple.getSecond().contains("rdf-syntax-ns#type")) {
                    relationPrediction.add(triple.getSecond());
                }
            }
        }
        Evaluator.addEntityLinkingSample(new ArrayList<>(GoldenLinkingUtil.removeAngleBrackets(new HashSet<>(entityPrediction))), entityGolden);
        Evaluator.addRelationLinkingSample(new ArrayList<>(GoldenLinkingUtil.removeAngleBrackets(new HashSet<>(relationPrediction))), relationGolden);
        Evaluator.addTypeLinkingSample(new ArrayList<>(GoldenLinkingUtil.removeAngleBrackets(new HashSet<>(typePrediction))), typeGolden);
    }

    /**
     * Detect entity for one node, fill the content of nodeEMap, nodeEListMap and noEntityNodeIds
     *
     * @param node descriptive node
     */
    public void initialEntityDetection(Node node) {
        int nodeID = node.getNodeID();

        //key: entity mention; value: List<Link>
        Map<String, List<Link>> entityLinkingMap = Detector.entityDetection(node.getStr(), globalEntityLinkMap.getData(),
                QAArgs.getGoldenLinkingMode(), GoldenLinkingUtil.getGoldenEntityLinkBySerialNumber(serialNumber));

        if (!entityLinkingMap.isEmpty()) {

            // if more than one entity mention is recognized, reduce them
            if (entityLinkingMap.keySet().size() > 1) {
                List<TwoTuple<String, Double>> scoreList = new ArrayList<>(); // pair (mention, score)

                List<Link> candidateUriList = entityLinkingMap.values().stream().flatMap(Collection::stream).collect(Collectors.toList());
                Set<String> oneHopProperties = KBUtil.getOneHopPotentialEntity(candidateUriList);

                for (String key : entityLinkingMap.keySet()) {
                    // use one-hop property to choose the appropriate topic mention

                    double score = SimilarityUtil.getMentionConfidence(key, entityLinkingMap);
                    for (String oneHopProperty : oneHopProperties) {
                        if (oneHopProperty.toLowerCase().contains(key.toLowerCase())
                                || (key.toLowerCase().contains(oneHopProperty.toLowerCase()) && oneHopProperty.length() >= 3)
                                || SimilarityUtil.getScoreIgnoreCase(key, oneHopProperty) > 0.9) {
                            score *= 0;
                            break;
                        }
                    }
                    scoreList.add(new TwoTuple<>(key, score));

                }

                // in descending order of confidence
                scoreList.sort((tuple1, tuple2) -> {
                    if (tuple2.getSecond() > tuple1.getSecond()) {
                        return 1;
                    } else if (tuple2.getSecond() < tuple1.getSecond()) {
                        return -1;
                    } else { // equal, reserve the longer
                        return tuple2.getFirst().length() - tuple1.getFirst().length();
                    }
                });

                String topicMention = scoreList.get(0).getFirst(); // choose the top-1 node as the topic entity
                List<Link> eLinkList = entityLinkingMap.get(topicMention);
                nodeEntityListMap.put(nodeID, eLinkList);

                //Clear other entities identified in EMAP Mention 
                for (int i = 1; i < scoreList.size(); i++) {
                    entityLinkingMap.remove(scoreList.get(i).getFirst());
                }
            } else {
                List<Link> eLinkList = entityLinkingMap.values().stream().flatMap(Collection::stream).collect(Collectors.toList());
                nodeEntityListMap.put(nodeID, eLinkList);
            }
        } else { // if entity linking is empty
            nodeIdsWithNoEntity.add(nodeID); //No Entity, join noentityNodeID 
        }
    }

    /**
     * Cut the Entity candidate (Entity disambiguation) through the synergy between each node
     *
     * @param relatedDesIDs DesnodeIDS related to current Entity
     */
    public void entityReduce(Set<Integer> relatedDesIDs) {
        Set<Integer> tempSet = new HashSet<>(nodeEntityListMap.keySet());
        tempSet.retainAll(relatedDesIDs);
        int eNodeNums = tempSet.size(); //Node containing entity detection results 

        if (eNodeNums >= 2) {
            //There are more than two description containing Entity 
            // description Union disambiguation 
            for (Integer id : tempSet) {

                List<Link> eLinkList = nodeEntityListMap.get(id);
                List<Link> otherList = new ArrayList<>(); //All physical link results of other Node 
                for (Integer otherId : nodeEntityListMap.keySet()) {
                    if (!otherId.equals(id)) {
                        otherList.addAll(nodeEntityListMap.get(otherId));
                    }
                }

                //Other lists have been emptied, no need to empty the current list 
                if (otherList.isEmpty()) {
                    continue;
                }

                Iterator<Link> iter = eLinkList.iterator();
                while (iter.hasNext()) {
                    Link link = iter.next();
                    boolean in2Hop = false;
                    for (Link otLink : otherList) {// Limit the distance between each entry does not exceed two hops 
                        if (KBUtil.isEPairInTwoHop(link.getUri(), otLink.getUri())) {
                            in2Hop = true;
                            break;
                        }
                    }
                    if (!in2Hop) {//The result of the link between other nodes is not in two, delete the current node 
                        iter.remove();
                    }
                }

            }
        } else {
            for (Integer id : tempSet) {
                List<Link> eLinkList = nodeEntityListMap.get(id);
                while (eLinkList.size() > 2) {//Leave only TOP2 entity link results 
                    eLinkList.remove(2);
                }

                // if the first link result is long and highly possible, only retain one
                Link firstLink = eLinkList.get(0);
                if (firstLink.getScore() > 0.9 && KBUtil.queryLabel(firstLink.getUri()).length() > 30) {
                    if (eLinkList.size() == 2) {
                        eLinkList.remove(1);
                    }
                }

                // if a linked entity is a title,e.g. "mayor of paris", replace with its instances
                List<Link> newLink = new ArrayList<>();
                Iterator<Link> iterator = eLinkList.iterator();
                while (iterator.hasNext()) {
                    Link eLink = iterator.next();
                    if (KBUtil.isaTitle(eLink.getUri())) {
                        iterator.remove();
                        List<String> titleInstances = KBUtil.getTitleInstances(eLink.getUri());
                        titleInstances.forEach(uri -> newLink.add(new Link(eLink.getMention(), uri, LinkEnum.ENTITY, eLink.getScore())));
                    }
                }
                if (!newLink.isEmpty()) {
                    eLinkList.clear();
                    eLinkList.addAll(newLink);
                }
            }

        }
    }

    /**
     * Preliminary RelationDetection for Node, populate NODERMAP and NODERLISTMAP
     *
     * @param node descriptive node
     */
    public void initialRelationDetection(Node node) {

        int nodeID = node.getNodeID();
        //Map<String, List<Link>> eMap = nodeEntityMap.getOrDefault(nodeID, new HashMap<>());
        List<Link> eLinkList = nodeEntityListMap.getOrDefault(nodeID, new ArrayList<>());
        //Replace the Entity in NodeStr to transform in <E0>, EntityIndexmap store <E0, DBR: OBAMA> such mapping 
        Map<String, String> entityIndexMap = Detector.replaceEntityInNode(node, eLinkList);

        // if it is the target entity, considering the trigger of the question
        Trigger trigger = Trigger.UNKNOWN;
        if (edg.findEntityBlockID(node) == 0) {
            trigger = edg.getTrigger();
        }

        Map<String, List<Link>> rMap = Detector.relationDetection(node.getStr(), entityIndexMap,
                eLinkList, trigger, QAArgs.getGoldenLinkingMode(),
                GoldenLinkingUtil.getGoldenRelationLinkBySerialNumber(serialNumber), this.globalRelationLinkMap);


        List<Link> rLinkList = rMap.values().stream().flatMap(Collection::stream).collect(Collectors.toList());

        if (!rLinkList.isEmpty()) {
            nodeRelationListMap.put(nodeID, rLinkList);
        }

    }

    /**
     * Splicing each of the Entity-containing DESNODE, generating candidate Threetuple <E, R, SCORE>
     *
     * @param node          desNode
     * @param relatedDesIDS NodeID with DESNODE related to current Entity
     * @return Current candidate for DesNode Threetuple <E, R, Score>
     */
    public List<ThreeTuple<String, String, Double>> nodeTupleGeneration(Node node, Set<Integer> relatedDesIDS) {
        int nodeID = node.getNodeID();

        List<Link> entityLinkList = nodeEntityListMap.getOrDefault(nodeID, null);
        List<Link> relationLinkList = nodeRelationListMap.getOrDefault(nodeID, null);
        List<ThreeTuple<String, String, Double>> tupleList = new ArrayList<>();

        if (entityLinkList != null) {//Current node has physical link results 
            if (relationLinkList == null) {//Current Node No RLINKLIST 
                //May be a Type misunderstanding as Entity 
                //Delete the result of entity identification 
                if (!NLPUtil.judgeIfEntity(node.getStr())) {//EqualNode in the Judge type question 
                    nodeEntityListMap.remove(node.getNodeID());
                }
            } else { // if relation link is not null
                HashSet<String> oneHopProperty = new HashSet<>();
                for (Link elink : entityLinkList) {
                    oneHopProperty.addAll(Detector.oneHopPropertyFiltered(elink.getUri()));
                }

                List<Link> otherRLinkList = new ArrayList<>();
                List<String> otherNodeStrs = new ArrayList<>();
                for (Integer id : nodeIdsWithNoEntity) {
                    if (relatedDesIDS.contains(id) && nodeRelationListMap.containsKey(id)) {
                        otherRLinkList.addAll(nodeRelationListMap.get(id));
                        otherNodeStrs.add(edg.getNodes()[id].getStr());
                    }
                }

                for (Link otherRLink : otherRLinkList) {
                    if (oneHopProperty.contains(otherRLink.getUri())) {
                        //If a relationship link of an Entity's Node is in another Node, add another Node RLIST 
                        // other node relation link result in oneHopProperty, add to relationLinkList
                        otherRLink.setScore(0.6);//Improve score 
                        relationLinkList.add(otherRLink);
                    }
                }

                Map<String, List<String>> labelUriMap = new HashMap<>();
                for (String prop : oneHopProperty) {
                    String propLabel = KBUtil.queryLabel(prop);
                    if (propLabel == null) {// no label, remove
                        propLabel = UriUtil.extractUri(prop);
                    }
                    if (!labelUriMap.containsKey(propLabel)) {
                        labelUriMap.put(propLabel, new ArrayList<>());
                    }
                    labelUriMap.get(propLabel).add(prop);
                }

                for (String otherStr : otherNodeStrs) {
                    HashMap<String, Double> labelSim = SimilarityUtil.getCompositeSetSimilarity(otherStr, labelUriMap.keySet());
                    for (String propLabel : labelSim.keySet()) {
                        double score = labelSim.get(propLabel);
                        if (score > 0.6) {
                            labelUriMap.get(propLabel).forEach(property ->
                                    relationLinkList.add(new Link(otherStr, property, LinkEnum.RELATION, score)));
                        }
                    }
                }


                LinkingTool.reSortRelationLinkList(relationLinkList); //Reorder rlinklist 

                for (Link entityLink : entityLinkList) {
                    for (Link relationLink : relationLinkList) {
                        if (KBUtil.isERexists(entityLink.getUri(), relationLink.getUri())) {
                            double score = entityLink.getScore() * relationLink.getScore();
                            tupleList.add(new ThreeTuple<>(entityLink.getUri(), relationLink.getUri(), score));
                        }
                    }
                }

                //Current Node generated tuPleList 
                //Reverse order 
                Ordering<ThreeTuple> threeTupleOrdering = new Ordering<ThreeTuple>() {
                    @Override
                    public int compare(@Nullable ThreeTuple t1, @Nullable ThreeTuple t2) {
                        if (t1 == null && t2 == null) return 0;
                        if (t1 == null) return 1;
                        if (t2 == null) return -1;
                        return Double.compare((double) t2.getThird(), (double) t1.getThird());
                    }
                };
                tupleList.sort(threeTupleOrdering);

                // Tuple cuts the score below a certain threshold
                if (tupleList.size() > 1 || edg.findEntityBlockID(node) != 0) {
                    tupleList.removeIf(o -> o.getThird() <= 0.05);
                }

                /*//Control the number of Ertuples no more than 5 
                while (tupleList.size() > 5) {
                    tupleList.remove(5);
                }*/

            }
        }
        return tupleList;
    }

    /**
     * Generate a Block Candidate Tuples
     *
     * @param relatedDesIDs RelatedDesids within this block
     * @return ¸ÃblockµÄcandidate block tuples
     */
    public List<TwoTuple<List<ThreeTuple<String, String, Double>>, Double>> blockTupleGeneration
    (Set<Integer> relatedDesIDs) {
        //Several Node stitching 
        List<TwoTuple<List<ThreeTuple<String, String, Double>>, Double>> candidateTuples = new ArrayList<>();

        // get each node's highest score
        LinkedHashMap<Integer, Double> nodeERScoreMap = new LinkedHashMap<>();
        for (Integer nodeId : nodeERTupleListMap.keySet()) {
            if (relatedDesIDs.contains(nodeId)) {
                List<ThreeTuple<String, String, Double>> tupleList = nodeERTupleListMap.get(nodeId);
                if (!tupleList.isEmpty()) {
                    nodeERScoreMap.put(nodeId, tupleList.get(0).getThird());
                } else {
                    nodeERScoreMap.put(nodeId, 0.0);
                }
            }
        }

        // sort node by score DESC
        List<Integer> nodeIDList = new ArrayList<>(nodeERScoreMap.keySet());
        nodeIDList.sort((o1, o2) -> Double.compare(nodeERScoreMap.get(o2), nodeERScoreMap.get(o1)));


        // There is current Block Node 
        if (!nodeIDList.isEmpty()) {
            for (Integer nodeID : nodeIDList) {
                List<ThreeTuple<String, String, Double>> nodeTuples = nodeERTupleListMap.get(nodeID);

                if (candidateTuples.isEmpty()) {//First addition, join all 
                    for (ThreeTuple<String, String, Double> tuple : nodeTuples) {
                        List<ThreeTuple<String, String, Double>> candidate = new ArrayList<>();
                        candidate.add(tuple);
                        candidateTuples.add(new TwoTuple<>(candidate, tuple.getThird()));
                    }
                } else {//There have been results, pay 
                    Iterator<TwoTuple<List<ThreeTuple<String, String, Double>>, Double>> iterator = candidateTuples.iterator();
                    List<TwoTuple<List<ThreeTuple<String, String, Double>>, Double>> newCandidateTuples = new ArrayList<>();
                    while (iterator.hasNext()) {
                        TwoTuple<List<ThreeTuple<String, String, Double>>, Double> oldTuple = iterator.next();
                        List<ThreeTuple<String, String, Double>> candidate = oldTuple.getFirst();
                        Double score = oldTuple.getSecond();

                        boolean added = false;
                        for (ThreeTuple<String, String, Double> tuple : nodeTuples) {
                            //Add new statement 
                            List<ThreeTuple<String, String, Double>> tempList = new ArrayList<>(candidate);
                            tempList.add(tuple);
                            if (KBUtil.isThreeTupleListExists(tempList)) {
                                newCandidateTuples.add(new TwoTuple<>(tempList, score * tuple.getThird()));
                                added = true;
                            }
                        }

                    }

                    //Generated new longer sparql, abandoning the original sparql, otherwise retaining the original Sparql 
                    if (!newCandidateTuples.isEmpty()) {
                        if (!QAArgs.isEvaluateCandNum()) {
                            candidateTuples = newCandidateTuples; //Update CandidateTuples 
                        } else {
                            // evaluate Candidates, reverse the old sparqls
                            candidateTuples.addAll(newCandidateTuples);
                        }
                    }
                }
            }
        }

        //Reverse order 
        candidateTuples.sort((o1, o2) -> Double.compare(o2.getSecond(), o1.getSecond()));

        int blockBound = QAArgs.getBlockSparqlNumUpperB();
        if (!QAArgs.isEvaluateCandNum()) {
            //Keep TOP 5 
            while (candidateTuples.size() > blockBound) {
                candidateTuples.remove(blockBound);
            }
        }

        //Reduce according to threshold 
        Iterator<TwoTuple<List<ThreeTuple<String, String, Double>>, Double>> iterator = candidateTuples.iterator();
        while (iterator.hasNext()) {
            double score = iterator.next().getSecond();
            if (score <= 0.05) {
                iterator.remove();
            }
        }

        return candidateTuples;
    }

    /**
     * Generate a Block Candidate Sparql List
     *
     * @param entityID      entityID(blockID)
     * @param edg           edg
     * @param relatedDesIDs RelatedDesids within this block
     * @return This Block Candidate Sparql List
     */
    public List<SparqlGenerator> blockSparqlGeneration(int entityID, EDG edg, Set<Integer> relatedDesIDs) {

        List<SparqlGenerator> sparqlList = new ArrayList<>();

        List<TwoTuple<List<ThreeTuple<String, String, Double>>, Double>> candidateTuples = blockCandidateMap.get(entityID);

        SparqlGenerator tmpSparqlGen = new SparqlGenerator();
        String finalVarName = "e" + entityID;
        tmpSparqlGen.setFinalVarName(finalVarName);

        boolean isRootJudge = false; //Whether it is rootentity and is Judge type 
        if (entityID == 0) {//Root Entity, generate sparql according to QuestionType 
            QueryType queryType = edg.getNodes()[0].getQueryType();

            if (queryType == QueryType.JUDGE) {
                SparqlGenerator sparqlGen = new SparqlGenerator(tmpSparqlGen);
                sparqlGen.setQuesType(queryType); // set the question type
                for (Integer nodeID : relatedDesIDs) {
                    if (edg.isEqualNode(nodeID)) {//Find Equal 
                        List<Link> nodeEList = nodeEntityListMap.get(nodeID);
                        if (nodeEList != null && !nodeEList.isEmpty()) {
                            sparqlGen.getVarValueMap().put(finalVarName, "<" + nodeEList.get(0).getUri() + ">");
                        }
                    } else {//Non-equal 
                        List<ThreeTuple<String, String, Double>> threeTuples = nodeERTupleListMap.get(nodeID);
                        //Join only the first TUPLE 
                        if (threeTuples != null && !threeTuples.isEmpty()) {
                            ThreeTuple<String, String, Double> threeTuple = threeTuples.get(0);
                            //sparqlGen.addTriple("?" + finalVarName, "?p", "<" + threeTuple.getFirst() + ">");//Adjacent 
                            sparqlGen.addTriple("?" + finalVarName, "<" + threeTuple.getSecond() + ">", "<" + threeTuple.getFirst() + ">");
                            sparqlGen.setScore(1.0);
                        }
                    }
                }
                sparqlList.add(sparqlGen);
                isRootJudge = true;

            } else { // if it's not judge type
                if (queryType == QueryType.COUNT) {//Set the question type to count 
                    tmpSparqlGen.setQuesType(QueryType.COUNT);
                } else {
                    tmpSparqlGen.setQuesType(QueryType.COMMON);
                }
            }
        }

        if (!isRootJudge) {// not root judge
            for (TwoTuple<List<ThreeTuple<String, String, Double>>, Double> candidate : candidateTuples) {
                SparqlGenerator sparqlGenerator = new SparqlGenerator(tmpSparqlGen); // copy construction
                List<ThreeTuple<String, String, Double>> tupleList = candidate.getFirst();
                sparqlGenerator.addTupleList(tupleList, "?" + finalVarName);
                sparqlGenerator.setScore(candidate.getSecond());// set the score
                sparqlList.add(sparqlGenerator);
            }
        }

        return sparqlList;
    }

}
