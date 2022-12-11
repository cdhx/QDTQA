package cn.edu.nju.ws.edgqa.domain.edg;

import cn.edu.nju.ws.edgqa.domain.beans.TreeNode;
import cn.edu.nju.ws.edgqa.domain.staticres.WordList;
import cn.edu.nju.ws.edgqa.utils.NLPUtil;
import cn.edu.nju.ws.edgqa.main.QAArgs;
import cn.edu.nju.ws.edgqa.utils.enumerates.DatasetEnum;
import cn.edu.nju.ws.edgqa.utils.enumerates.KBEnum;
import cn.edu.nju.ws.edgqa.utils.enumerates.QueryType;
import cn.edu.nju.ws.edgqa.utils.enumerates.Trigger;
import cn.edu.nju.ws.edgqa.utils.linking.LinkingTool;
import org.json.JSONArray;
import org.json.JSONObject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.IntStream;

import static cn.edu.nju.ws.edgqa.domain.beans.TreeNode.*;
import static cn.edu.nju.ws.edgqa.domain.staticres.WordList.*;


/**
 * KB-independent Entity Description Graph for Question Understanding
 */
public class EDG {
    /**
     * logger
     */
    private static final Logger logger = LoggerFactory.getLogger(EDG.class);
    /**
     * The maximum EDG node number.
     */
    private static final int MAX_SIZE = 20;
    /**
     * the target KB, DBpedia by default
     */
    private static KBEnum KB = KBEnum.DBpedia;
    /**
     * The EDG node array.
     */
    private Node[] nodes;

    /**
     * The EDG edge matrix.
     */
    private Edge[][] edges;

    /**
     * The number of EDG node.
     */
    private int numNode;

    /**
     * The number of entities to be retrieved, i.e., the number of entity blocks
     */
    private int numEntity;

    /**
     * The original natural language question.
     */
    private String question;

    /**
     * The string of syntax treeNode.
     */
    private String syntaxTreeText;

    /**
     * The custom syntax treeNode
     */
    private TreeNode syntaxTreeNode;

    /**
     * The question with long entities replaced.
     */
    private String taggedQuestion;

    /**
     * The map of long entity substitutions, e.g., <e1>: Obama
     */
    private Map<String, String> entityMap;

    /**
     * Default constructor
     */
    private EDG() {
        initializeMembers();
    }

    /**
     * Constructor with a question
     *
     * @param question the natural language question
     */
    public EDG(String question) {
        this();
        try {
            construct(question);
        } catch (Exception e) {
            constructFlattenEDG(question);
        }
    }

    /**
     * Constructor with a question and the potential entity map
     *
     * @param question  the natural language question
     * @param entityMap the potential entity map, key:<e0> value:Obama
     */
    public EDG(String question, Map<String, String> entityMap) {
        this();
        this.entityMap = entityMap;
        construct(question);
    }

    public static KBEnum getKB() {
        return KB;
    }

    /**
     * construct EDG from a json decomposition by neural model
     *
     * @param o json decomposition
     * @return the edg constructed
     */
    public static EDG fromDecomposition(JSONObject o) {
        EDG edg = new EDG();
        edg.question = o.getString("corrected_question");
        //edg.question = o.getString("question");

        // Common root by default
        Node rootNode = edg.createNode(Node.NODE_TYPE_ROOT);

        String intent = o.getString("intent");
        if (intent.equals("ASK")) {
            rootNode.setQuesType(QueryType.JUDGE);
        } else if (intent.equals("COUNT")) {
            rootNode.setQuesType(QueryType.COUNT);
        } else {
            rootNode.setQuesType(QueryType.COMMON);
        }

        // create the first entityNode
        Node entityNode1 = edg.createNode(Node.NODE_TYPE_ENTITY);
        entityNode1.setEntityID(0);
        edg.edges[rootNode.getNodeID()][entityNode1.getNodeID()].setEdgeType(Edge.TYPE_QUEST);
        edg.numEntity = 1;

        Node entityNode2 = null;
        // create the second entityNode if necessary
        if (o.keySet().contains("E2")) {
            entityNode2 = edg.createNode(Node.NODE_TYPE_ENTITY);
            entityNode2.setEntityID(1);
            edg.numEntity = 2;
        }

        List<String> descriptions1 = new ArrayList<>();
        if (o.keySet().contains("E1")) {
            Object e1 = o.get("E1");
            if (e1 instanceof JSONArray) {
                for (int i = 0; i < ((JSONArray) e1).length(); i++) {
                    descriptions1.add(((JSONArray) e1).getString(i));
                }
            } else if (e1 instanceof String) {
                descriptions1.add((String) e1);
            }
        } else { // no decomposition, make the whole question as a description
            descriptions1.add(edg.question);
        }

        for (String des : descriptions1) {

            Node desNode = edg.createNode(Node.NODE_TYPE_VERB);
            desNode.setStr(des);
            desNode.setOriginStr(des);
            desNode.setEntityID(entityNode1.getEntityID());
            edg.edges[entityNode1.getNodeID()][desNode.getNodeID()].setEdgeType(Edge.TYPE_VERB);

            if (des.contains("#E2")) {
                desNode.setContainsRefer(true);
                edg.edges[desNode.getNodeID()][entityNode2.getNodeID()].setEdgeType(Edge.TYPE_REFER);
            }
        }

        List<String> descriptions2 = new ArrayList<>();
        if (o.keySet().contains("E2")) {
            JSONArray e2 = o.getJSONArray("E2");
            for (int i = 0; i < e2.length(); i++) {
                descriptions2.add(e2.getString(i));
            }
        }

        for (String des : descriptions2) {
            Node desNode = edg.createNode(Node.NODE_TYPE_VERB);
            desNode.setStr(des);
            desNode.setOriginStr(des);
            desNode.setEntityID(entityNode2.getEntityID());
            edg.edges[entityNode2.getNodeID()][desNode.getNodeID()].setEdgeType(Edge.TYPE_VERB);
        }

        return edg;

    }

    /**
     * Judge if a node has any referred entity
     *
     * @param edg       the whole EDG
     * @param nodeIndex index of a node
     * @return true if one of the neighbor of this node is a referred entity
     */
    public static boolean judgeRefer(EDG edg, int nodeIndex) {
        if (nodeIndex >= edg.getNumNode()) { // index out of bound
            return false;
        }
        Edge[][] edges = edg.getEdges();
        for (int i = 0; i < edg.getNumNode(); i++) {
            if (edges[nodeIndex][i].edgeType == Edge.TYPE_REFER) { // if any edge of this node is type refer
                return true;
            }
        }
        return false;
    }

    /**
     * parse JSONObject to EDG instance
     *
     * @param edgJSON JSONObject of EDG
     * @return EDG instance
     */
    public static EDG fromJSON(JSONObject edgJSON) {
        EDG edg = new EDG();
        edg.question = edgJSON.getString("question");
        edg.taggedQuestion = edgJSON.getString("taggedQuestion");
        edg.syntaxTreeText = edgJSON.getString("syntaxTreeText");
        edg.syntaxTreeNode = createTree(edg.syntaxTreeText);
        edg.numNode = edgJSON.getInt("nodeNum");
        edg.numEntity = edgJSON.getInt("entityNum");
        edg.entityMap = new HashMap<>();
        JSONObject entityMapJSON = edgJSON.getJSONObject("entityMap");
        for (String key : entityMapJSON.keySet()) {
            edg.entityMap.put(key, entityMapJSON.getString(key));
        }
        edg.nodes = new Node[edg.numNode];
        JSONArray nodesJSON = edgJSON.getJSONArray("nodes");
        for (int i = 0; i < nodesJSON.length(); i++) {
            edg.nodes[i] = Node.fromJSON(nodesJSON.getJSONObject(i));
        }
        edg.edges = new Edge[edg.numNode][edg.numNode];
        for (int i = 0; i < edg.numNode; i++){      //Need to be initialized manually
            for(int j = 0; j < edg.numNode; j++){   //changed on 7.20
                edg.edges[i][j] = new Edge();
            }
        }
        System.out.print(edg.edges[0][0].edgeType);
        JSONArray edgesJSON = edgJSON.getJSONArray("edges");
        for (int i = 0; i < edgesJSON.length(); i++) {
            JSONObject edgeJSON = edgesJSON.getJSONObject(i);
            int from = edgeJSON.getInt("from");
            int to = edgeJSON.getInt("to");
            edg.edges[from][to] = Edge.fromJSON(edgeJSON);
        }

        return edg;
    }

    public static void init(DatasetEnum dataset) {
        System.out.println("[INFO] Setting EDG KB..");
        switch (dataset) {
            case LC_QUAD:
            case QALD_9:
            case UNKNOWN: {
                KB = KBEnum.DBpedia;
                break;
            }
        }
        System.out.println("[INFO] EDG KB set");
    }

    /**
     * Calculate the start and the end of a node in the sentence
     *
     * @param node     a description node in EDG
     * @param treeNode the syntactic tree corresponding to this description node
     */
    public static void ComputeNodeStartEnd(Node node, TreeNode treeNode) {
        node.setStart(TreeNode.getFirstLeaf(treeNode).index);//set start
        node.setEnd(getLastLeaf(treeNode).index + 1);//set end
        if (getFirstLeaf(treeNode).index <= node.getStart() && getLastLeaf(treeNode).index >= node.getEnd() - 1) {
            String tempString = TreeNode.selectLeafByIndex(treeNode, node.getStart(), node.getEnd());
            tempString = tempString.replace("-LRB-", "(").replace("-RRB-", ")");
            node.setStr(tempString);
        }

    }

    /**
     * Determine a prepositional phrase contain only one preposition
     * e.g., What country is Mount Everest in?
     *
     * @param treeNode a syntax tree
     * @return if the PP in the syntax tree contains only one preposition, true; false otherwise
     */
    public static boolean IsPPOnlyIN(TreeNode treeNode) {
        if ((treeNode.data.equals("SQ ")) && treeNode.children.size() == 3 && treeNode.children.get(2).data.equals("PP ")) {
            treeNode = treeNode.children.get(2);
            if (treeNode.children.size() == 1) {
                if (treeNode.getFirstChild().data.startsWith("IN")) {
                    return true;
                }
            }
        } else if (treeNode.data.equals("PP ") && treeNode.children.size() == 1 && treeNode.children.get(0).data.startsWith("IN")) {

            return true;
        } else if (treeNode.data.equals("VP ")) {
            if (treeNode.children.size() > 1) {
                if (treeNode.children.get(1).data.equals("PP ")) {
                    return IsPPOnlyIN(treeNode.children.get(1));
                }
            }
        }
        return false;
    }

    /**
     * get the query type
     *
     * @return the query type of current question
     */
    public QueryType getQueryType() {
        if (nodes[0].getQueryType() != null)
            return nodes[0].getQueryType();
        return QueryType.UNKNOWN;
    }

    /**
     * get the trigger
     *
     * @return the trigger of current question
     */
    public Trigger getTrigger() {
        if (nodes[0].getTrigger() == null) {
            return Trigger.UNKNOWN;
        }
        if ("where".equals(nodes[0].getTrigger().toLowerCase().trim())) {
            return Trigger.WHERE;
        } else if ("when".equals(nodes[0].getTrigger().toLowerCase().trim())) {
            return Trigger.WHEN;
        } else if ("who".equals(nodes[0].getTrigger().toLowerCase().trim())) {
            return Trigger.WHO;
        } else if (nodes[0].getTrigger().trim().toLowerCase().startsWith("how")) {
            return Trigger.HOW;
        } else if (nodes[0].getQueryType() == QueryType.JUDGE) {
            return Trigger.IS;
        }
        return Trigger.UNKNOWN;
    }

    /**
     * Serialize the EDG to json
     *
     * @return the serialized json object
     */
    public JSONObject toJSON() {
        JSONObject edgJSON = new JSONObject();
        edgJSON.put("question", question);
        edgJSON.put("taggedQuestion", taggedQuestion);
        edgJSON.put("syntaxTreeText", syntaxTreeText);
        edgJSON.put("nodeNum", numNode);
        edgJSON.put("entityNum", numEntity);

        JSONObject entityMapJSON = new JSONObject();
        for (String key : entityMap.keySet()) {
            entityMapJSON.put(key, entityMap.get(key));
        }
        edgJSON.put("entityMap", entityMapJSON);

        JSONArray nodesJSON = new JSONArray();
        for (int i = 0; i < numNode; i++) {
            if (nodes[i].getNodeType() > 0) {
                nodesJSON.put(nodes[i].toJSON());
            }
        }

        edgJSON.put("nodes", nodesJSON);
        JSONArray edgesJSON = new JSONArray();
        for (int i = 0; i < numNode; i++) {
            for (int j = 0; j < numNode; j++) {
                edges[i][j].from = i;
                edges[i][j].to = j;
                if (edges[i][j].edgeType > 0) {
                    edgesJSON.put(edges[i][j].toJSON());
                }
            }
        }
        edgJSON.put("edges", edgesJSON);

        return edgJSON;
    }

    public Map<String, String> getEntityMap() {
        return entityMap;
    }

    public Node[] getNodes() {
        return nodes;
    }

    public Edge[][] getEdges() {
        return edges;
    }

    public int getNumNode() {
        return numNode;
    }

    public String getQuestion() {
        return question;
    }

    public String getSyntaxTreeText() {
        return syntaxTreeText;
    }

    public TreeNode getSyntaxTree() {
        return syntaxTreeNode;
    }

    public int getNumEntity() {
        return numEntity;
    }

    /**
     * EDG string including nodes, edges and long entities
     *
     * @return EDG string
     */
    @Override
    public String toString() {
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append("Question:").append(question).append("\n");
        stringBuilder.append("NodeNum:").append(numNode).append("\n");

        int edgeNum = 0;
        for (int i = 0; i < numNode; i++) {
            for (int j = 0; j < numNode; j++) {
                if (edges[i][j] != null && edges[i][j].edgeType > 0) {
                    edgeNum++;
                }
            }
        }
        stringBuilder.append("EdgeNum:").append(edgeNum).append("\n");
        stringBuilder.append("Nodes:\n");
        for (int i = 0; i < numNode; i++) {
            stringBuilder.append(nodes[i].toString()).append("\n");
        }
        stringBuilder.append("Edges:\n");
        for (int i = 0; i < numNode; i++) {
            for (int j = 0; j < numNode; j++) {
                if (edges[i][j] != null && edges[i][j].edgeType > 0) {
                    edges[i][j].setTo(j);
                    edges[i][j].setFrom(i);
                    stringBuilder.append(edges[i][j].toString()).append("\n");
                }
            }
        }
        stringBuilder.append("LongEntities: ");
        stringBuilder.append(entityMap.toString()).append("\n");
        return stringBuilder.toString();
    }

    private void initializeMembers() {
        nodes = new Node[MAX_SIZE];                         // node array
        edges = new Edge[MAX_SIZE][MAX_SIZE];               // edge matrix
        for (int i = 0; i < edges.length; i++) {            //initialize NoEdge
            for (int j = 0; j < edges[0].length; j++) {
                edges[i][j] = new Edge();
            }
        }

        numNode = 0;                                        // the number of nodes
        numEntity = 0;                                      // number of entity blocks
        question = null;                                    // the natural language question
        syntaxTreeText = null;                              // the text of syntax treeNode
        syntaxTreeNode = null;                                  // syntax treeNode
        entityMap = new HashMap<>();                        // long entity map
    }

    private void constructFlattenEDG(String question) {
        // default constructor failed, return a default one
        initializeMembers();
        this.question = question;
        // question type recognition
        QueryType questionType = getCoarseQuestionType(question);
        // create nodes
        Node rootNode = createNode(Node.NODE_TYPE_ROOT);
        rootNode.setQuesType(questionType);
        Node entityNode = createNode(Node.NODE_TYPE_ENTITY);
        entityNode.setEntityID(0);
        Node desNode = createNode(Node.NODE_TYPE_VERB);
        desNode.setStr(question);
        desNode.setEntityID(0);
        numEntity = 1;

        // create edges
        edges[rootNode.getNodeID()][entityNode.getNodeID()].setEdgeType(Edge.TYPE_QUEST);
        edges[entityNode.getNodeID()][desNode.getNodeID()].setEdgeType(Edge.TYPE_VERB);
    }

    private void construct(String question) {
        // origin question
        System.out.println("Original Question: " + question);
        // pre process the question
        this.question = Preprocessor.preProcessQuestion(question);
        System.out.println("Question after preprocess: " + this.question);

        // recognize long entity
        if (this.entityMap == null) {
            this.entityMap = new LinkedHashMap<>(); // key:<e1> value: Obama
        }
        taggedQuestion = LinkingTool.recognizeLongEntity(this.question, this.entityMap, EDG.KB);

        System.out.println("TaggedQuestion: " + taggedQuestion);

        // generate syntax treeNode of taggedQuestion
        this.syntaxTreeText = NLPUtil.getSyntaxTree(taggedQuestion);
        this.syntaxTreeNode = createTree(this.syntaxTreeText);
        System.out.println(NLPUtil.transferParentheses(this.syntaxTreeText));

        // parse the syntaxTree, generate nodes and edges
        Sent(this.syntaxTreeNode);

        //count the number of entityBlock
        CountEntityBlock();
        //generate nodeTree for each description node
        GenerateNodeTree();
        //generate nodeStr for each description node
        GenerateString();
        //replace refer span by #entity{i}
        ProcessRefer();
        //some post processing
        PostProcess();
        //long entity back replace
        BackReplaceLongE();
        //trim nodeStr
        TrimNode();
        //final check
        FinalCheck();
        // make a copy of node str
        BackupNodeStr();

    }

    private void BackupNodeStr() {
        for (int i = 0; i < numNode; i++) {
            Node node = nodes[i];
            if (node.getNodeType() >= 3) {
                node.setOriginStr(node.getStr());
            }
        }
    }

    private void FinalCheck() {
        if (numNode <= 2) {
            constructFlattenEDG(this.question);
        }
    }

    /**
     * substitute irregular expressions in the node
     */
    private void TrimNode() {

        for (int i = 0; i < numNode; i++) {

            if (nodes[i].getNodeType() >= Node.NODE_TYPE_VERB) {
                String nodeStr = nodes[i].getStr();
                // processing some special symbols
                nodeStr = nodeStr.replace(" 's", "'s")
                        .replace(" (", "(")
                        .replace(" )", ")")
                        .replace(" - ", "-")
                        .replaceAll("\\?$", "")
                        .replaceAll("\\.$", "")
                        .replaceAll(",$", "")
                        .replaceAll("!$", "")
                        .trim();

                nodeStr = nodeStr.replace("-LRB- ", "(")
                        .replace(" -RRB-", ")").trim();

                // processing wrong identification due to syntax tree wrong identification for list query
                Matcher matcher = Pattern.compile("(?i)(^(list) (.*))").matcher(nodeStr);
                if (matcher.matches()) {
                    nodeStr = matcher.group(3).trim();
                }

                // processing redundant introductory words of subordinate clauses
                nodeStr = nodeStr.replaceAll("^(which|where|who) ", "");

                // processing redundant spaces
                nodeStr = nodeStr.replaceAll(" +", " ");

                nodes[i].setStr(nodeStr);
            }
        }

    }

    /**
     * count the number of entityBlock
     */
    private void CountEntityBlock() {
        numEntity = 0;
        for (int i = 1; i < numNode; i++) {
            if (nodes[i].getNodeType() == Node.NODE_TYPE_ENTITY) {
                nodes[i].setEntityID(numEntity); // set the entityID of entity node
                for (int j = 1; j < numNode; j++) {
                    if (edges[i][j].edgeType > Edge.TYPE_REFER) {
                        nodes[j].setEntityID(numEntity); // set the entityID of each related node
                    }
                }
                numEntity++;
            }
        }
    }

    /**
     * set the syntaxTree of each description Node
     */
    private void GenerateNodeTree() {
        for (int nodeIdx = 0; nodeIdx < numNode; nodeIdx++) {
            if (nodes[nodeIdx].getNodeType() == Node.NODE_TYPE_VERB || nodes[nodeIdx].getNodeType() == Node.NODE_TYPE_NON_VERB) {
                nodes[nodeIdx].setTree(TreeNode.getAncestor(syntaxTreeNode, nodes[nodeIdx].getStart(), nodes[nodeIdx].getEnd()));
                while (nodes[nodeIdx].getTree() != null && nodes[nodeIdx].getTree().children.size() == 1) { // if node has only one child
                    nodes[nodeIdx].setTree(nodes[nodeIdx].getTree().children.get(0));
                }
            }
        }
    }

    /**
     * Generate string for each EDG node
     */
    private void GenerateString() {
        for (int i = 0; i < numNode; i++) {
            if (nodes[i].getNodeType() == Node.NODE_TYPE_VERB || nodes[i].getNodeType() == Node.NODE_TYPE_NON_VERB) {
                //already has nodeStr, pass
                if (nodes[i].getStr() != null && !nodes[i].getStr().equals("")) {
                    continue;
                }

                StringBuilder t = new StringBuilder();
                for (int j = nodes[i].getStart(); j < nodes[i].getEnd(); j++) {
                    t.append(Objects.requireNonNull(getLeaf(syntaxTreeNode, j)).str);
                }
                if (t.toString().trim().equals("")) {
                    nodes[i].setStr("");
                } else {
                    nodes[i].setStr(t.toString().trim());
                }
            }
        }
    }

    /**
     * replace the referring span by #entity{ID}, set the containRefer=true as well
     */
    private void ProcessRefer() {
        for (int i = 0; i < numNode; i++) {
            for (int j = 0; j < numNode; j++) {
                if (edges[i][j].edgeType == 2) {
                    Edge referEdge = edges[i][j];
                    int start = referEdge.getStart();
                    int end = referEdge.getEnd();

                    Node node = nodes[i];
                    String nodeStr = node.getStr();
                    node.setContainsRefer(true);
                    ArrayList<String> tokens = NLPUtil.getTokens(nodeStr);

                    int entityID = nodes[j].getEntityID();
                    int relStart = start - node.getStart();
                    int relEnd = relStart + end - start;

                    if (relStart < tokens.size() && relEnd >= 0 && relStart >= 0) { // prevent out of index
                        List<String> replaceTokens = tokens.subList(relStart, Math.min(relEnd, tokens.size()));
                        StringBuilder sb = new StringBuilder();
                        for (String token : replaceTokens) {
                            sb.append(token).append(" ");
                        }
                        node.setStr(nodeStr.replace(sb.toString().trim(), "#entity" + entityID));
                    }
                }
            }
        }

    }

    /**
     * Replace long entities
     */
    private void BackReplaceLongE() {
        Map<String, String> toReplace = entityMap;

        // back to syntax tree
        LinkedList<TreeNode> toSearch = new LinkedList<>();
        toSearch.push(syntaxTreeNode);
        while (!toSearch.isEmpty()) {
            TreeNode poll = toSearch.poll();
            if (poll.isLeaf()) {
                if (toReplace.containsKey(poll.str.trim())) {
                    poll.str = " " + toReplace.get(poll.str.trim());
                }
            } else {
                for (TreeNode child : poll.children) {
                    toSearch.push(child);
                }
            }
        }

        // node string
        if (!entityMap.keySet().isEmpty()) {
            for (int i = 0; i < numNode; i++) {
                Node node = nodes[i];
                if (node.getNodeType() >= 3) {
                    for (String key : entityMap.keySet()) {
                        node.setStr(node.getStr().replace(key, entityMap.get(key)));
                    }
                }
            }
        }
    }

    /**
     * post process, including implicit entity detection, conjunction process, remove redundant refer, etc
     */
    private void PostProcess() {

        // process implicit entity to be referred
        ProcessImplicitRefer();

        // process conjunction
        ProcessConjunction();

        // process redundant refer
        ProcessRedundantRefer();

        // count the number of entity blocks
        CountEntityBlock();

        // process useless description nodes
        ProcessUselessDesNode();
    }

    /**
     * process implicit entity to be referred
     */
    private void ProcessImplicitRefer() {

        int curNodeNum = numNode;

        for (int i = 0; i < curNodeNum; i++) {
            Node node = nodes[i];
            if (node.getNodeType() >= Node.NODE_TYPE_VERB) {

                //remove some punctuations
                String tempString = node.getStr().trim()
                        .replaceAll("\\?$", "")
                        .replaceAll("\\.$", "")
                        .replaceAll("!$", "")
                        .replaceAll(" 's", "'s")
                        .replaceAll("\\( ", "(")
                        .replaceAll(" \\)", ")").trim();
                node.setStr(tempString);
                String nodeStr = node.getStr();

                if (KB != KBEnum.Freebase) { // only for DBpedia
                    //some regex rules to detect implicit entity to be referred

                    //apposite led by comma
                    Matcher matcher1 = Pattern.compile("(?i)((.*), (which|who|where|whose) (.*))").matcher(nodeStr);
                    if (matcher1.matches()) {
                        System.out.println("[DEBUG] Matching Matcher1:" + nodeStr);

                        //split a desNode
                        Node desNode = createNode(Node.NODE_TYPE_VERB);
                        desNode.setStr(matcher1.group(4));
                        edges[findEntityNodeID(node)][desNode.getNodeID()].setEdgeType(Edge.TYPE_VERB);

                        //modify the origin nodeStr
                        node.setStr(matcher1.group(2));

                    }

                    //(the xxx) whose xx's aaa is bbb
                    Matcher matcher2 = Pattern.compile("(?i)((.*)whose (.*)('s|s') ((.*) (is|was|were|are) (.*)))").matcher(nodeStr);
                    if (matcher2.matches()) {
                        System.out.println("[DEBUG] Matching Matcher2:" + nodeStr);

                        //create a new entity, set the refer edge
                        Node entityNode = createNode(Node.NODE_TYPE_ENTITY);
                        edges[node.getNodeID()][entityNode.getNodeID()].setEdgeType(Edge.TYPE_REFER);
                        edges[node.getNodeID()][entityNode.getNodeID()].setEnd(node.getEnd());
                        edges[node.getNodeID()][entityNode.getNodeID()].setStart(node.getEnd() - matcher2.group(5).trim().split(" ").length);

                        //create a new desNode
                        Node desNode = createNode(Node.NODE_TYPE_VERB);
                        desNode.setStr(matcher2.group(5).trim());
                        desNode.setEnd(node.getEnd());
                        desNode.setStart(node.getEnd() - matcher2.group(5).trim().split(" ").length);
                        edges[entityNode.getNodeID()][desNode.getNodeID()].setEdgeType(Edge.TYPE_VERB);

                        //create another desNode(usually type)
                        if (matcher2.group(2) != null && !matcher2.group(2).trim().equals("")) {
                            Node desNode1 = createNode(Node.NODE_TYPE_NON_VERB);
                            desNode1.setStr(matcher2.group(2));
                            desNode1.setStart(node.getStart());
                            desNode1.setEnd(node.getStart() + matcher2.group(2).split(" ").length);
                            edges[findEntityNodeID(node)][desNode1.getNodeID()].setEdgeType(Edge.TYPE_NON_VERB);
                        }

                        //modify the origin nodeStr and containRefer tag
                        node.setStr(matcher2.group(3).trim() + " is #entity" + entityNode.getEntityID());
                        node.setContainsRefer(true);

                        continue;
                    }

                    // e.g. "acted in the movie which was directed by Quentin Tarantino"
                    Matcher matcher3 = Pattern.compile("(?i)((.*)((the|a|an) (.*) (whose|who|which|where|that) (.*)))").matcher(nodeStr);
                    if (matcher3.matches()) {

                        System.out.println("[DEBUG] Matching Matcher3:" + nodeStr);

                        System.out.println(matcher3.group(7));
                        //the xxx of xxx which is also xxx
                        if (matcher3.group(7).matches("^(is|was|are|were) also (.*)")
                                || matcher3.group(7).contains(" and ")) {//conjunction,turn to conjunction module, pass
                            System.out.println("[DEBUG] Matcher3: contain conjunction, pass");
                            continue;
                        } else if (matcher3.group(7).contains("#entity")) {//already contains refer, pass
                            System.out.println("[DEUBG] Matcher3: already contains Refer, pass");
                            continue;
                        }

                        if (!containsRelation(matcher3.group(2)) && containsRelation(matcher3.group(5))) {
                            // e.g. "where was the scientist born whose doctoral advisor is ..."

                            //find the type word in gropu5
                            List<String> tokens = NLPUtil.getTokens(matcher3.group(5));

                            String typeString = tokens.get(0);

                            //create entity node
                            Node entityNode = createNode(Node.NODE_TYPE_ENTITY);
                            edges[node.getNodeID()][entityNode.getNodeID()].setEdgeType(Edge.TYPE_REFER);

                            // modify origin nodeStr
                            node.setStr(matcher3.group(2) + " " + matcher3.group(5).replace(typeString, " #entity" + entityNode.getEntityID()));
                            node.setContainsRefer(true);

                            //description before "whose"
                            Node desNode1 = createNode(Node.NODE_TYPE_NON_VERB);
                            desNode1.setStr(typeString);
                            edges[entityNode.getNodeID()][desNode1.getNodeID()].setEdgeType(Edge.TYPE_NON_VERB);

                            //description after "whose"
                            Node desNode2 = createNode(Node.NODE_TYPE_VERB);
                            desNode2.setStr(matcher3.group(7).trim());
                            edges[entityNode.getNodeID()][desNode2.getNodeID()].setEdgeType(Edge.TYPE_VERB);

                        } else {
                            //create entity and refer edge
                            Node entityNode = createNode(Node.NODE_TYPE_ENTITY);
                            edges[node.getNodeID()][entityNode.getNodeID()].setEdgeType(Edge.TYPE_REFER);
                            node.setStr(matcher3.group(2).trim() + " #entity" + entityNode.getEntityID());
                            node.setContainsRefer(true);

                            // type description
                            Node desNode1 = createNode(Node.NODE_TYPE_NON_VERB);
                            desNode1.setStr(matcher3.group(5).trim());
                            edges[entityNode.getNodeID()][desNode1.getNodeID()].setEdgeType(Edge.TYPE_NON_VERB);

                            // description after "whose"
                            Node desNode2 = createNode(Node.NODE_TYPE_VERB);
                            desNode2.setStr(matcher3.group(7).trim());
                            edges[entityNode.getNodeID()][desNode2.getNodeID()].setEdgeType(Edge.TYPE_VERB);

                        }
                        continue;
                    }


                    // e.g. "the composer of <e0>"
                    Matcher matcher4 = Pattern.compile("(?i)((.*) (the (.*) of <e\\d>) (.*))").matcher(nodeStr);
                    if (matcher4.matches()) {
                        System.out.println("[DEBUG] Matching Matcher4:" + nodeStr);
                        String restStr = matcher4.group(2) + " " + matcher4.group(5);
                        if (containsRelation(restStr)) {
                            String newDesStr = matcher4.group(3);

                            //create entity Node
                            Node newEn = createNode(Node.NODE_TYPE_ENTITY);
                            edges[i][newEn.getNodeID()].setEdgeType(Edge.TYPE_REFER);

                            //create desNode
                            Node newDes = createNode(Node.NODE_TYPE_NON_VERB);
                            newDes.setStr(newDesStr);
                            edges[newEn.getNodeID()][newDes.getNodeID()].setEdgeType(Edge.TYPE_NON_VERB);

                            //modify the origin nodeStr
                            String newStr = matcher4.group(2) + " #entity" + findEntityBlockID(newEn) + " " + matcher4.group(5);
                            node.setStr(newStr);
                            node.setContainsRefer(true);

                        } else {
                            System.out.println("[DEBUG] Matcher4 : no relation, pass");
                        }
                        continue;
                    }

                    // e.g., the xx of the xx by|in|with xx
                    Matcher matcher5 = Pattern.compile("(.*)the (.*) of ((the )?(.*) (by|in|with) (.*))").matcher(nodeStr);
                    if (matcher5.matches()) {

                        System.out.println("[DEBUG] Matching Matcher5:" + nodeStr);
                        String referString = matcher5.group(3);

                        if (matcher5.group(5).contains(" and ") || matcher5.group(5).contains(" also ")) {// contains conjunction, pass
                            continue;
                        }

                        if (NLPUtil.judgeIfEntity(matcher5.group(5))) {// is an entity
                            continue;
                        }

                        //create entity node
                        Node entityNode = createNode(Node.NODE_TYPE_ENTITY);
                        edges[node.getNodeID()][entityNode.getNodeID()].setEdgeType(Edge.TYPE_REFER);

                        // modify origin nodeStr
                        node.setStr(nodeStr.replace(referString, "#entity" + entityNode.getEntityID()));
                        node.setContainsRefer(true);

                        // create desNode for the new entity
                        TreeNode newTreeNode = createTree(NLPUtil.getSyntaxTree(referString)).getFirstChild();
                        if (newTreeNode.getData().trim().equals("S") && newTreeNode.getChildren().size() == 2) {
                            // S->NP VP
                            // e.g. "the song written by .."
                            TreeNode npTreeNode = newTreeNode.getFirstChild();
                            TreeNode vpTreeNode = newTreeNode.getLastChild();

                            Node desNode1 = createNode(Node.NODE_TYPE_NON_VERB);
                            desNode1.setStr(selectLeaf(npTreeNode).trim());
                            edges[entityNode.getNodeID()][desNode1.getNodeID()].setEdgeType(Edge.TYPE_NON_VERB);

                            Node desNode2 = createNode(Node.NODE_TYPE_VERB);
                            desNode2.setStr(selectLeaf(vpTreeNode).trim());
                            edges[entityNode.getNodeID()][desNode2.getNodeID()].setEdgeType(Edge.TYPE_VERB);

                        } else { // not S, the whole sentence as
                            Node desNode = createNode(Node.NODE_TYPE_NON_VERB);
                            desNode.setStr(referString);
                            edges[entityNode.getNodeID()][desNode.getNodeID()].setEdgeType(Edge.TYPE_NON_VERB);
                        }
                        continue;
                    }

                    // e.g., the father of the prime minister of UK
                    Matcher matcher6 = Pattern.compile("(.*)the (.*) of (the (.*) of (.*))").matcher(nodeStr);
                    if (matcher6.matches()) { // it contains hidden entities, an reference is needed

                        System.out.println("[DEBUG] Matcher6:" + nodeStr);

                        //the xx of the xx and xx of ...
                        // 'and' is wrongly identified, continue directly, and the conjunction processor will deal with it
                        if (matcher6.group(4).contains("and ") || matcher6.group(4).contains("also ")) {
                            System.out.println("Matcher0: conjunctions detected");
                            continue;
                        }

                        String newDesStr = matcher6.group(3);
                        // create a new entity, and set an reference edge
                        Node newEn = createNode(Node.NODE_TYPE_ENTITY);
                        edges[i][newEn.getNodeID()].setEdgeType(Edge.TYPE_REFER);

                        // create a new description and set NVP edge
                        Node newDes = createNode(Node.NODE_TYPE_NON_VERB);
                        newDes.setStr(newDesStr);
                        edges[newEn.getNodeID()][newDes.getNodeID()].setEdgeType(Edge.TYPE_NON_VERB);

                        // modify the entity to be referred in the original node, and set containRefer to true
                        String newStr = nodeStr.replace(newDesStr, "#entity" + findEntityBlockID(newEn));
                        node.setStr(newStr);
                        node.setContainsRefer(true);

                        continue;
                    }

                    // wrote the subsequent work of ...
                    Matcher matcher7 = Pattern.compile("(.*) (the (.*) (of|at|in|by) (.*))").matcher(nodeStr);
                    if (matcher7.matches()) {
                        String relWord = matcher7.group(1); // preceding spans that may contain a relation
                        String newDesStr = matcher7.group(2); // following entities to be referred

                        if (newDesStr.contains(" and ")) {
                            continue;
                        }

                        if (relWord.contains("one of") || relWord.contains("member of") || beWordSet.contains(relWord.trim())) { // one of|member of|is|was misidentification
                            continue;
                        }

                        if (NLPUtil.judgeIfEntity(newDesStr)) { // it's already a named entity, pass
                            continue;
                        }

                        // the xxx of the <e0> do
                        if (newDesStr.matches("(.*) (<e\\d>) (.*)")) {
                            Matcher matcher = Pattern.compile("((.*) <e\\d>) (.*)").matcher(newDesStr);
                            if (matcher.matches()) {
                                String group1 = matcher.group(1);
                                String group2 = matcher.group(2);

                                String typeString = null;
                                String relString = null;

                                // the river starting from <e0> flow through
                                if (group2.startsWith("the")) {
                                    if (group2.trim().equals("the")) {
                                        continue;
                                    }
                                    ArrayList<String> tokens = NLPUtil.getTokens(group2);
                                    if (tokens.size() >= 4 && (tokens.get(2).endsWith("ing") || tokens.get(2).endsWith("ed"))) { // the river starting at <e0>
                                        typeString = tokens.get(0) + " " + tokens.get(1);
                                        relString = group1.replaceAll(typeString, "").trim();
                                    } else {
                                        relString = group1;
                                    }
                                }

                                // create a new entity
                                Node newEn = createNode(Node.NODE_TYPE_ENTITY);
                                edges[i][newEn.getNodeID()].setEdgeType(Edge.TYPE_REFER);

                                // create a new description
                                if (typeString != null) { // there is type
                                    Node desNode1 = createNode(Node.NODE_TYPE_NON_VERB);
                                    desNode1.setStr(typeString);
                                    edges[newEn.getNodeID()][desNode1.getNodeID()].setEdgeType(Edge.TYPE_NON_VERB);
                                }

                                Node desNode2 = createNode(Node.NODE_TYPE_VERB);
                                desNode2.setStr(relString);
                                edges[newEn.getNodeID()][desNode2.getNodeID()].setEdgeType(Edge.TYPE_VERB);

                                // modify the entity to be referred in the original node
                                String newStr = nodeStr.replace(group1, "#entity" + findEntityBlockID(newEn));
                                node.setStr(newStr);
                                node.setContainsRefer(true);
                                continue;
                            }

                        }

                        if (judgeIfAuxiliary(relWord)) {// sentences guided by auxiliary verbs, e.g., did the director of XX win

                            // find the NP node
                            TreeNode nodeTree;
                            if (node.getTree() != null) {
                                nodeTree = node.getTree();
                            } else {
                                nodeTree = createTree(NLPUtil.getSyntaxTree(nodeStr)).getFirstChild();
                            }
                            int treeID = 0;
                            for (; treeID < nodeTree.getChildren().size(); treeID++) {
                                if (nodeTree.getChildren().get(treeID).getData().trim().equals("NP")) {
                                    break;
                                }
                            }

                            TreeNode npTreeNode = null;
                            // no NP found
                            if (treeID == nodeTree.getChildren().size()) {
                                npTreeNode = nodeTree.getLastChild();
                            } else {
                                npTreeNode = nodeTree.getChildren().get(treeID);
                            }
                            String desStr;

                            // create a new entity
                            Node newEn = createNode(Node.NODE_TYPE_ENTITY);
                            edges[i][newEn.getNodeID()].setEdgeType(Edge.TYPE_REFER);
                            edges[i][newEn.getNodeID()].setStart(getFirstLeaf(npTreeNode).index);

                            // VBD+NP, did [the director of xx win] is taken as a NP, verbs need to be extracted
                            if (treeID == nodeTree.getChildren().size() - 1) {
                                edges[i][newEn.getNodeID()].setEnd(getLastLeaf(npTreeNode).index);
                                desStr = selectLeafByIndex(npTreeNode, getFirstLeaf(npTreeNode).index, getLastLeaf(npTreeNode).index);
                            } else {
                                edges[i][newEn.getNodeID()].setEnd(getLastLeaf(npTreeNode).index + 1);
                                desStr = selectLeaf(npTreeNode);
                            }

                            // create a new description and set a NVP edge
                            Node newDes = createNode(Node.NODE_TYPE_NON_VERB);
                            newDes.setStr(desStr);
                            newDes.setStart(edges[i][newEn.getNodeID()].getStart());
                            newDes.setEnd(edges[i][newEn.getNodeID()].getEnd());

                            edges[newEn.getNodeID()][newDes.getNodeID()].setEdgeType(Edge.TYPE_NON_VERB);


                            // modify the entity to be referred in the original node
                            String newStr = nodeStr.replace(newDes.getStr(), "#entity" + findEntityBlockID(newEn));
                            node.setStr(newStr);
                            node.setContainsRefer(true);

                            continue;

                        } else if (containsRelation(relWord)) { // matcher2.group(1) contains relation, it needs to be split


                            // create a new entity
                            Node newEn = createNode(Node.NODE_TYPE_ENTITY);
                            edges[i][newEn.getNodeID()].setEdgeType(Edge.TYPE_REFER);

                            // create a new description, and set NVP edge
                            Node newDes = createNode(Node.NODE_TYPE_NON_VERB);
                            newDes.setStr(newDesStr);
                            edges[newEn.getNodeID()][newDes.getNodeID()].setEdgeType(Edge.TYPE_NON_VERB);

                            // modify the entity to be referred in the original node
                            String newStr = relWord.trim() + " #entity" + findEntityBlockID(newEn);
                            node.setStr(newStr);
                            node.setContainsRefer(true);

                            continue;

                        } else {
                            System.out.println("[DEBUG] there's no relation, Matcher2 is passed");
                        }
                    }

                    // (is xxx) (with xxx as xx)
                    Matcher matcher8 = Pattern.compile("((is|was|are|were) (.*)) with (.*)").matcher(nodeStr);
                    if (matcher8.matches()) {

                        System.out.println("[DEBUG] Matcher8:" + nodeStr);

                        if (!NLPUtil.judgeIfVP(matcher8.group(3))) { // prevent the case such as "is associated with"

                            // split the description
                            String newDesStr = matcher8.group(4);
                            Node newDes = createNode(Node.NODE_TYPE_NON_VERB);
                            newDes.setStr(newDesStr);
                            edges[findEntityNodeID(node)][newDes.getNodeID()].setEdgeType(Node.NODE_TYPE_NON_VERB);

                            nodeStr = matcher8.group(1);
                            node.setStr(nodeStr);
                        }
                        continue;

                    }
                }
            }

        }

    }

    /**
     * Post-processing unnecessary description nodes
     */
    private void ProcessUselessDesNode() {
        for (int i = 0; i < numNode; i++) {
            if (nodes[i] != null && nodes[i].getNodeType() >= Node.NODE_TYPE_VERB) {
                String nodeStr = nodes[i].getStr();
                if (nodeStr != null && nodeStr.trim().toLowerCase().matches("^(there|here|a|an|the|all|one of)$")) {
                    // meaningless one word, delete this node
                    deleteNode(i);
                    i--; // the node has been removed, i moves forward 1 position
                }
            }
        }
    }

    /**
     * Post-processing unnecessary reference edge in EDG
     */
    private void ProcessRedundantRefer() {

        for (int i = 0; i < numNode; i++) {
            Node node = nodes[i];
            if (node.getNodeType() >= Node.NODE_TYPE_VERB) { // it's a description node
                if (node.isContainsRefer()) { // this description node has an reference edge
                    String nodeStr = node.getStr().trim();
                    if (nodeStr.matches("(^#entity\\d$)|(^(is|was|are|were|did|does|do)( also)? #entity\\d$)")
                            || nodeStr.matches("^((is|are|was|were) )?the ((total|whole) )?(name|number|count) of #entity\\d$")
                            || EDG.getKB() == KBEnum.Freebase) {

                        if (nodeStr.matches("^((is|are|was|were) )?the ((total|whole) )?(number|count) of #entity\\d$")) {
                            nodes[0].setQuesType(QueryType.COUNT);
                        }

                        // it only contains this referred entity, or the situation like "is #entity1..."
                        int referredEntityNodeID = 0;
                        // find the nodeID of this referring entity
                        for (int j = 0; j < numNode; j++) {
                            if (edges[i][j] != null && edges[i][j].getEdgeType() == Edge.TYPE_REFER) { // find the referring entity
                                referredEntityNodeID = j;
                                break;
                            }
                        }
                        int originEntityNodeID = findEntityNodeID(node);

                        if (referredEntityNodeID > 0) { // find the nodeID of the referring entity

                            // the entityID to be removed
                            int oldEntityID = nodes[referredEntityNodeID].getEntityID();

                            // remove the current description first
                            int nodeID = deleteNode(node.getNodeID());
                            if (referredEntityNodeID > nodeID) { // referred entity is after nodeID
                                referredEntityNodeID--;
                            }

                            // merge the referring entity and the original entity
                            for (int j = 0; j < numNode; j++) {
                                Edge curEdge = edges[referredEntityNodeID][j];
                                if (curEdge != null && curEdge.getEdgeType() > Edge.TYPE_NO_EDGE) {
                                    edges[originEntityNodeID][j] = curEdge; // the edge moves
                                    edges[referredEntityNodeID][j] = new Edge(); // the original edge is reset
                                }

                                // another direction
                                Edge curEdge1 = edges[j][referredEntityNodeID];
                                if (curEdge1 != null && curEdge1.getEdgeType() > Edge.TYPE_NO_EDGE) {
                                    edges[j][originEntityNodeID] = curEdge1;
                                    edges[j][referredEntityNodeID] = new Edge(); // the original edge is reset
                                }
                            }
                            deleteNode(referredEntityNodeID);

                            // processing #entity{i} in description nodes
                            for (int j = 0; j < numNode; j++) {
                                if (nodes[j].getNodeType() >= Node.NODE_TYPE_VERB) {
                                    String nodeString = nodes[j].getStr();
                                    Matcher matcher = Pattern.compile("#entity(\\d)").matcher(nodeString);
                                    if (matcher.find()) {
                                        String toReplace = matcher.group();
                                        int oldID = Integer.parseInt(matcher.group(1));
                                        if (oldID >= oldEntityID) {
                                            int newID = oldID - 1;
                                            nodeString = nodeString.replace(toReplace, "#entity" + newID);
                                            nodes[j].setStr(nodeString);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }


    }

    /**
     * Post-processing, dealing with 'and' 'also' conjunction that are not partitioned in the EDG
     */
    private void ProcessConjunction() {

        int curNodeNum = numNode;
        for (int nodeIdx = 0; nodeIdx < curNodeNum; nodeIdx++) {
            Node node = nodes[nodeIdx];
            int entityNodeID = findEntityNodeID(node);
            if ((node.getNodeType() == Node.NODE_TYPE_VERB || node.getNodeType() == Node.NODE_TYPE_NON_VERB)) {

                if (!node.isContainsRefer()) {
                    // VP-des or NVP-des without refer edge
                    String nodeStr = node.getStr();
                    if (nodeStr != null && (nodeStr.contains(" and ") || nodeStr.contains(" also ") || nodeStr.contains(" both ") || nodeStr.contains(" & "))) {
                        TreeNode treeNode = node.getTree();
                        if (treeNode == null) {
                            treeNode = createTree(NLPUtil.getSyntaxTree(node.getStr()));
                        }
                        // both ... and ..., remove both
                        if (nodeStr.matches("(.*) both (.*) and (.*)")) {
                            nodeStr = nodeStr.replace("both", "");
                            node.setStr(nodeStr);
                            treeNode = createTree(NLPUtil.getSyntaxTree(nodeStr));
                        }

                        // ... and ... both ...
                        Matcher matcher = Pattern.compile("(.*) and (.*) both (.*)").matcher(nodeStr);
                        if (matcher.matches()) {
                            String former = matcher.group(1).trim();
                            String latter = matcher.group(2).trim();
                            String relStr = matcher.group(3).trim();
                            node.setStr(former + " " + relStr);

                            Node newVPNode = createNode(Node.NODE_TYPE_VERB);
                            newVPNode.setStr(latter + " " + relStr);
                            edges[entityNodeID][newVPNode.getNodeID()].setEdgeType(Edge.TYPE_VERB);

                            // continue next Node
                            continue;
                        }

                        // replace 'and also' to 'and'
                        nodeStr = nodeStr.replace("and also ", "and ");
                        node.setStr(nodeStr);

                        // and conjunction
                        if (nodeStr.contains("and ") || nodeStr.contains("& ")) {
                            System.out.println("split Node By CC:" + nodeStr);
                            splitNodeByCC(treeNode, nodeIdx);
                        }
                        if (node.getStr().contains("also ")) {
                            nodeStr = node.getStr().replaceAll("(am|is|are|was|were|have been|has been|has|have) also", "also").trim();

                            String[] str = nodeStr.split("also");
                            System.out.println("Split By also:" + Arrays.toString(str));
                            if (str.length >= 2) {
                                String former = str[0].trim();
                                String latter = str[1].trim();
                                int formerLen = NLPUtil.getTokens(former).size();
                                int latterLen = NLPUtil.getTokens(latter).size();
                                if (!former.equals("") && !latter.equals("")) {// also is not at the start, additional nodes are generated
                                    node.setStr(former);
                                    if (NLPUtil.judgeIfVP(former)) {
                                        if (node.getNodeType() != 3) {
                                            edges[entityNodeID][node.getNodeID()].edgeType = 3;
                                        }
                                    }

                                    // set newNode
                                    Node newNode = new Node();
                                    newNode.setStr(latter);
                                    newNode.setNodeID(numNode);
                                    newNode.setEntityID(node.getEntityID());

                                    // set start and end
                                    newNode.setEnd(node.getEnd());
                                    newNode.setStart(newNode.getEnd() - latterLen);
                                    node.setEnd(node.getStart() + formerLen);

                                    if (NLPUtil.judgeIfVP(latter)) {
                                        newNode.setNodeType(Node.NODE_TYPE_VERB);
                                    } else {
                                        newNode.setNodeType(Node.NODE_TYPE_NON_VERB);
                                    }
                                    nodes[numNode] = newNode;
                                    numNode++;

                                    Edge newEdge = new Edge();
                                    newEdge.edgeType = newNode.getNodeType() == 3 ? 3 : 4;
                                    edges[entityNodeID][newNode.getNodeID()] = newEdge;
                                } else { // if former == "" or latter == "", "also" has no meaning
                                    node.setStr(nodeStr.replace("also ", ""));
                                }
                            }
                        }
                    }
                } else {
                    // VP-des or NVP-des contains refer edge
                    String nodeStr = node.getStr();
                    Matcher matcher1 = Pattern.compile("^(.*?)((is|are|was|were|and) )?(also|and) ((is|are|was|were) )?(#entity\\d)$").matcher(nodeStr);
                    if (matcher1.matches()) {
                        // create a new description and connect it with the current node
                        Node desNode = createNode(Node.NODE_TYPE_VERB);
                        desNode.setStr(matcher1.group(1).trim());
                        desNode.setEntityID(findEntityBlockID(nodes[entityNodeID]));
                        desNode.setStart(node.getStart());
                        desNode.setEnd(desNode.getStr().split(" ").length + desNode.getStart());
                        edges[entityNodeID][desNode.getNodeID()].setEdgeType(Edge.TYPE_VERB);

                        // modify the current nodeStr
                        node.setStr(matcher1.group(7));
                        node.setStart(desNode.getEnd());

                    }


                }
            }
        }
    }

    /**
     * Divide the description into two parts based on conjunctions and generate additional descriptions
     *
     * @param nodeTree  the nodeTree of node to be split
     * @param nodeIndex the nodeID of node to be split
     */
    private void splitNodeByCC(TreeNode nodeTree, int nodeIndex) {


        LinkedList<TreeNode> toSearch = new LinkedList<>();
        toSearch.add(nodeTree);
        int splitIndex = -1;

        // search for tag "CC"
        while (!toSearch.isEmpty()) {
            TreeNode cur = toSearch.pop();
            if (cur.data.trim().equals("CC")) {
                splitIndex = cur.index;
                break;
            } else {
                if (!cur.children.isEmpty()) {
                    toSearch.addAll(cur.children);
                }
            }
        }

        // found the conjunction node tagged "CC"
        if (splitIndex >= 0) {
            TreeNode ccNode = getLeaf(nodeTree, splitIndex);
            TreeNode parentNode = ccNode.parent;

            // start and end index for this node
            int nodeStart = TreeNode.getFirstLeaf(nodeTree).index;
            int nodeEnd = TreeNode.getLastLeaf(nodeTree).index;

            //startIndex for the former part
            int firstStart = getFirstLeaf(parentNode).index;

            //relation before "and" (e.g. written A and B)
            StringBuilder sb = new StringBuilder();
            for (int j = nodeStart; j < firstStart; j++) {
                sb.append(Objects.requireNonNull(getLeaf(nodeTree, j)).str);
            }

            String commonRelationBefore = sb.toString().trim();
            System.out.println("commonRelation Before:" + commonRelationBefore);

            //relation after "and" (e.g. "A and B shares common...")
            StringBuilder sb1 = new StringBuilder();
            if (!containsRelation(commonRelationBefore)) {// no commonRelation before;
                if (parentNode.data.trim().equals("NP") || parentNode.data.trim().equals("NML")) {// e.g. "A and B are both famous"
                    List<TreeNode> siblings = parentNode.getParent().getChildren();
                    int indexOfParent = siblings.indexOf(parentNode);
                    for (int i = indexOfParent + 1; i < siblings.size(); i++) {// father's sibling nodes
                        sb1.append(" ").append(selectLeaf(siblings.get(i)));
                    }
                }
            }
            String commonRelationAfter = sb1.toString().trim();
            System.out.println("commonRelation After:" + commonRelationAfter);


            // split the node by ccNode
            StringBuilder former = new StringBuilder();
            for (int i = firstStart; i < splitIndex; i++) {
                former.append(getLeaf(nodeTree, i).str.trim()).append(" ");
            }

            StringBuilder poster = new StringBuilder();
            for (int i = splitIndex + 1; i <= nodeEnd; i++) {
                poster.append(getLeaf(nodeTree, i).str.trim()).append(" ");
            }

            // trim string
            String formerString = former.toString().replace("also ", "").trim();
            String posterString = poster.toString().replace("also ", "").trim();

            // whether formerString/posterString contains relation
            boolean formerRelationTag = containsRelation(formerString);
            boolean posterRelationTag = containsRelation(posterString);

            // whether commonRelationBefore/commonRelationAfter contains relation
            boolean commonRelationBeforeTag = containsRelation(commonRelationBefore);
            boolean commonRleationAfterTag = containsRelation(commonRelationAfter);

            // concatenate
            if (!formerRelationTag || !posterRelationTag) {
                if (!formerRelationTag) {// no relation in former string
                    if (commonRelationBeforeTag) {
                        formerString = (commonRelationBefore + " " + formerString).trim();
                    } else if (commonRleationAfterTag) {
                        formerString = (formerString + " " + commonRelationAfter).trim();
                    }
                }
                if (!posterRelationTag) {// no relation in posterString
                    if (commonRleationAfterTag) {
                        posterString = (posterString + " " + commonRelationAfter).trim();
                    } else if (commonRelationBeforeTag) {
                        posterString = (commonRelationBefore + " " + posterString).trim();
                    }
                }

                if (!formerRelationTag && !posterRelationTag && !commonRelationBeforeTag && !commonRleationAfterTag) {
                    String nodeStr = nodes[nodeIndex].getStr();
                    String conjunctionSpan = selectLeaf(parentNode);
                    String commonRel = nodeStr.replace(conjunctionSpan, "").trim();

                    formerString = commonRel + " " + formerString;
                    posterString = commonRel + " " + posterString;

                }

            }
            System.out.println("former:" + formerString);
            System.out.println("poster:" + posterString);

            String des1 = formerString;
            String des2 = posterString;

            // origin node
            Node node = nodes[nodeIndex];
            node.setStr(des1);
            node.setStart(nodeStart);
            node.setEnd(splitIndex);

            int newNodeType;
            if (NLPUtil.judgeIfVP(des2)) { //VP node
                newNodeType = Node.NODE_TYPE_VERB;
            } else { //NVP node
                newNodeType = Node.NODE_TYPE_NON_VERB;
            }
            Node newNode = createNode(newNodeType);
            newNode.setEntityID(node.getEntityID());
            newNode.setStr(des2);
            newNode.setStart(splitIndex + 1);
            newNode.setEnd(nodeEnd);

            int entityNodeID = findEntityNodeID(node);
            Edge newEdge = edges[entityNodeID][newNode.getNodeID()];
            // edge type follow the node type
            newEdge.edgeType = newNode.getNodeType() == Node.NODE_TYPE_VERB ? Edge.TYPE_NON_VERB : Edge.TYPE_NON_VERB;

        }
    }

    /**
     * Find the nodeID of the connected entity of a description node
     *
     * @param node the node of the entity to be found
     * @return the nodeID of the entity connect to it
     */
    public int findEntityNodeID(Node node) {

        if (node.getNodeType() == Node.NODE_TYPE_ENTITY) {
            return node.getNodeID();
        }

        for (int i = 0; i < numNode; i++) {
            System.out.print("finding entity");
            System.out.print(i);
            if (nodes[i].getNodeType() == Node.NODE_TYPE_ENTITY && edges[i][node.getNodeID()].getEdgeType() > 0) { // it contains an edge
                return i;
            }
        }
        return 1;
    }

    /**
     * Find the blockID of a node
     *
     * @param node node
     * @return the blockID
     */
    public int findEntityBlockID(Node node) {
        if (node.getNodeType() == Node.NODE_TYPE_ROOT) { // root
            return 0;
        }
        if (node.getNodeType() == Node.NODE_TYPE_ENTITY) {
            int entityNum = 0;
            for (int i = 0; i < node.getNodeID(); i++) {
                if (nodes[i].getNodeType() == Node.NODE_TYPE_ENTITY) {
                    entityNum++;
                }
            }
            return entityNum;
        }
        if (node.getNodeType() >= Node.NODE_TYPE_VERB) {
            return findEntityBlockID(nodes[findEntityNodeID(node)]);
        }
        return 0;
    }

    /**
     * Processes syntactic tree nodes that have been determined to be attached to the node with a given entityIdx
     *
     * @param treeNode  the node to be connected in the syntax tree
     * @param entityIdx the nodeID of entity node
     * @param lenOfAdvp
     */
    private void ConcateDes(TreeNode treeNode, int entityIdx, int lenOfAdvp) {

        if (treeNode.data.trim().equals("NP")) { // Is+NP+NP
            List<Integer> listg = NP(treeNode, nodes[entityIdx]);
            if (listg.size() > 1) {
                logger.error("ConcateDes NP multiple nodes are generated:" + question);
            }
        } else if (treeNode.data.trim().equals("PP")) { // Is+NP+PP
            List<Integer> listg = PP(treeNode, nodes[entityIdx]);
            if (listg.size() > 1) {
                logger.error("ConcateDes PP multiple nodes are generated:" + question);
            }
        } else if (treeNode.data.trim().equals("VP")) { // Is + NP + VP
            List<Integer> listg = SQ(treeNode, nodes[entityIdx]);
            for (Integer in : listg) {
                if (nodes[entityIdx].getNodeType() == Node.NODE_TYPE_ENTITY) {
                    edges[entityIdx][in].edgeType = Edge.TYPE_VERB;
                }
                nodes[in].setStart(nodes[in].getStart() - lenOfAdvp);

            }
        } else if (treeNode.data.equals("ADVP ") || treeNode.data.equals("ADJP ")) {

            Node desNode = createNode(Node.NODE_TYPE_VERB);
            desNode.setStr(selectLeaf(treeNode).trim());
            desNode.setStart(getFirstLeaf(treeNode).index);
            desNode.setEnd(getLastLeaf(treeNode).index);
            edges[entityIdx][desNode.getNodeID()].setEdgeType(Edge.TYPE_VERB);

        } else if (treeNode.data.equals("S ") && treeNode.children.size() > 1 && treeNode.children.get(1).data.equals("VP ") && treeNode.children.get(0).data.equals("ADVP ")) {
            List<Integer> listg = S(treeNode, nodes[entityIdx]);
            for (Integer in : listg) {
                if (nodes[entityIdx].getNodeType() == Node.NODE_TYPE_ENTITY) {
                    edges[entityIdx][in].edgeType = Edge.TYPE_VERB;
                }
                nodes[in].setStart(nodes[in].getStart() - lenOfAdvp);

            }
        } else if (treeNode.data.trim().equals("S")) { // S is nested in SQ
            //(SQ (SQ (VBZ S(...))
            List<Integer> listg = S(treeNode, nodes[entityIdx]);
            for (Integer in : listg) {
                if (nodes[entityIdx].getNodeType() == Node.NODE_TYPE_ENTITY) {
                    edges[entityIdx][in].edgeType = Edge.TYPE_VERB;
                }
                nodes[in].setStart(nodes[in].getStart() - lenOfAdvp);
            }
        } else if (treeNode.data.trim().equals("SBAR")) {
            SQ(treeNode, nodes[entityIdx]);
        } else { // np by default
            List<Integer> listg = NP(treeNode, nodes[entityIdx]);
        }
    }

    /**
     * Determine whether a tree contains a subordinate clause
     *
     * @param treeNode the root node of a tree
     * @return true if it contains a subordinate clause; false otherwise
     */
    private boolean containClause(TreeNode treeNode) {
        if (treeNode.leaf) {
            return false;
        }

        for (TreeNode temp : treeNode.children) {

            if (temp.data.trim().startsWith("SBAR")) {
                return true;
            }
            if (temp.data.trim().startsWith("VP")) {
                return true;
            }
            if (containClause(temp)) {
                return true;
            }
        }

        return false;
    }

    /**
     * Process imperative sentences, get the nodeID of the entity node
     *
     * @param rootIdx  nodeID of root
     * @param trigger  trigger word
     * @param treeNode the root node of the syntax tree
     * @return the nodeID of the entity node
     */
    private int imperativeSentence(int rootIdx, String trigger, TreeNode treeNode) {

        // create a new entity node
        Node node = createNode(Node.NODE_TYPE_ENTITY);
        edges[rootIdx][node.getNodeID()].edgeType = Edge.TYPE_QUEST;

        // a tree to be added as a description
        LinkedList<TreeNode> toAdd = new LinkedList<>();

        // process some special sentences
        if (taggedQuestion.matches("(?i)(^(give|show|tell) me (a|the) (count|number) of (.*))")) {
            // regenerating the tree
            treeNode = createTree(NLPUtil.getSyntaxTree(
                    taggedQuestion.replaceAll("(?i)(^(give|show|tell) me (a|the) (count|number) of)", "").trim())).
                    getFirstChild();
            nodes[0].setQuesType(QueryType.COUNT);
        }


        for (int i = 0; i < treeNode.children.size(); i++) {
            TreeNode cur = treeNode.children.get(i);

            if (selectLeaf(cur).toLowerCase().trim().startsWith(trigger.trim())) { // a node with a trigger
                // sometimes the node of a trigger has sibling nodes, the siblings need to be added
                LinkedList<TreeNode> toSearch = new LinkedList<>();
                toSearch.add(cur);
                while (!toSearch.isEmpty()) {  // search for ancestors of the node where the trigger is located
                    TreeNode pop = toSearch.pop();
                    if (selectLeaf(pop).toLowerCase().trim().startsWith(trigger.trim())) { // the ancestor is found
                        if (!pop.children.isEmpty()) {
                            toSearch.addAll(pop.children);
                        }
                    } else if (!trigger.trim().contains(selectLeaf(cur).toLowerCase().trim())) {
                        // a node without trigger and not a part of the trigger
                        System.out.println("Added:" + selectLeaf(pop));
                        if (!pop.data.trim().equals("CC")) {
                            toAdd.add(pop);
                        }
                    }
                }
            } else if (!trigger.trim().contains(selectLeaf(cur).toLowerCase().trim())) {
                // a node without trigger
                // and the current node is not a part of the trigger, e.g., 'give' is a part of 'give me'
                if (!cur.data.trim().equals("CC")) { // not a conjunction such as 'and'
                    toAdd.add(cur);
                }
            }
        }

        for (TreeNode add : toAdd) {
            ConcateDes(add, node.getNodeID(), 0);

        }

        return node.getNodeID();
    }

    /**
     * Process general questions, and get the nodeID of the generated entity node
     *
     * @param listTreeNode the syntax node to be processed
     * @param lenOfAdvp
     * @return the nodeID of the generated entity node
     */
    private int GeneralQuestion(List<TreeNode> listTreeNode, int lenOfAdvp) { // the general questions

        // create a new entity node
        Node node = createNode(Node.NODE_TYPE_ENTITY);

        for (TreeNode treeNode : listTreeNode) {
            ConcateDes(treeNode, node.getNodeID(), lenOfAdvp);
        }

        return node.getNodeID();
    }

    /**
     * node2 is referred from node1, calculate the start and the end in node2 and set them to the edge.
     *
     * @param node1Idx node1 id
     * @param node2Idx node2 id
     * @param treeNode syntax tree for node2
     */
    private void ComputeEdgeStartEnd(int node1Idx, int node2Idx, TreeNode treeNode) {

        edges[node1Idx][node2Idx].setEdgeType(Edge.TYPE_REFER);
        edges[node1Idx][node2Idx].start = TreeNode.getFirstLeaf(treeNode).index;  // start
        edges[node1Idx][node2Idx].end = TreeNode.getLastLeaf(treeNode).index + 1; // end
        // [start, end)
    }

    /**
     * process the 'S' clause
     *
     * @param treeNode the syntax treeNode node tagged 'S'
     * @param node     entity node
     * @return the IDs of the nodes generated
     */
    private List<Integer> S(TreeNode treeNode, Node node) { // S phrase

        List<Integer> nodeIdxs = new ArrayList<>();
        if (treeNode.children.size() == 1) { // S has only one VP child
            TreeNode treeNode1 = treeNode.children.get(0); // the child of S
            if (treeNode1.data.trim().equals("VP")) {
                return SQ(treeNode1, node);
            } else {
                Node desNode = createNode(Node.NODE_TYPE_VERB);
                ComputeNodeStartEnd(desNode, treeNode1);
                if (node.getNodeType() == Node.NODE_TYPE_ENTITY) {
                    edges[node.getNodeID()][desNode.getNodeID()].setEdgeType(Edge.TYPE_VERB);
                } else {
                    int enNodeID = findEntityNodeID(node);
                    edges[enNodeID][desNode.getNodeID()].setEdgeType(Edge.TYPE_VERB);
                }
            }

        } else if (treeNode.children.size() == 2) { // S has two children
            TreeNode treeNode1 = treeNode.children.get(0); // the left child of S
            TreeNode treeNode2 = treeNode.children.get(1); // the right child of S

            Node node1 = new Node();// create a temp node
            if (node.getNodeType() == Node.NODE_TYPE_ENTITY) { // current node is an entity
                // create a VPNode
                node1 = createNode(Node.NODE_TYPE_VERB);
                nodeIdxs.add(node1.getNodeID());    // add it to the result list
                ComputeNodeStartEnd(node1, treeNode);
                edges[node.getNodeID()][node1.getNodeID()].edgeType = Edge.TYPE_VERB;// VP edge

            } else {
                node1 = node; // current node is not an entity, get node1 as current node
            }

            if (treeNode1.data.trim().equals("NP") && treeNode2.data.trim().equals("VP")) { // NP + VP

                if (containClause(treeNode1)) { // tree1 contains a dependent clause before generating a new node
                    // call NP function to process
                    NP(treeNode1, node1);
                }

                List<Integer> list = SQ(treeNode2, node1);
                for (Integer x : list) {
                    edges[node.getNodeID()][x].setEdgeType(Edge.TYPE_VERB);
                }

            } else if (treeNode1.data.trim().equals("ADVP") && treeNode2.data.trim().equals("VP")) {
                SQ(treeNode2, node1);
            }
        } else { // the number of children of S is greater than or equal to 3

            Node node1 = createNode(Node.NODE_TYPE_VERB);
            ComputeNodeStartEnd(node1, treeNode);
            if (node.getNodeType() == Node.NODE_TYPE_ENTITY) {
                edges[node.getNodeID()][node1.getNodeID()].setEdgeType(Edge.TYPE_VERB);
            }
            nodeIdxs.add(node1.getNodeID());

            for (int i = 0; i < treeNode.children.size(); i++) {

                TreeNode temp = treeNode.children.get(i);
                if (temp.data.trim().equals("NP")) {
                    if (containClause(temp)) {
                        NP(temp, node1);
                    }
                } else if (temp.data.trim().equals("VP")) {
                    SQ(temp, node1);
                } else if (temp.data.trim().equals("PP")) {
                    if (containClause(temp)) {
                        PP(temp, node1);
                    }
                }
            }
        }
        return nodeIdxs;
    }

    /**
     * process the 'SQ' clause
     *
     * @param treeNode the syntax treeNode node tagged 'SQ'
     * @param node     current EDG node
     * @return the IDs of the nodes generated
     */
    private List<Integer> SQ(TreeNode treeNode, Node node) { // S,SQ,VP

        List<Integer> listg = new ArrayList<>();
        int len = treeNode.children.size(); // the number of children of tree

        String s = "";
        if (len > 0) {
            s = selectLeaf(treeNode.children.get(len - 1), ""); // the string of the most right child of SQ
        } else { // current node is already a leaf node
            s = treeNode.getStr().trim();
        }

        int sign = 0; // whether it contains a punctuation
        if (s.trim().equals(".") || s.trim().equals("?")) {
            len--;
            sign = 1;
        }

        if (len == 1) { // only one child excluding punctuation
            if (treeNode.data.trim().equals("SQ")) { // tree is SQ
                if (!treeNode.children.get(0).data.trim().equals("VP")) { // SQ has only one child and it is not VP
                    logger.error("SQ not Cover-1:" + question);
                } else { // the VP node under SQ
                    TreeNode VPNode = treeNode.children.get(0);
                    return SQ(VPNode, node);
                }
            } else {

                Node entityNode = null;
                if (node.getNodeType() == Node.NODE_TYPE_ENTITY) {
                    entityNode = node;
                } else if (node.getNodeType() == Node.NODE_TYPE_ROOT) {
                    entityNode = nodes[node.getNodeID() + 1];
                }

                if (entityNode != null) {// only work when first visit
                    Node VPNode = createNode(Node.NODE_TYPE_VERB);
                    ComputeNodeStartEnd(VPNode, treeNode);
                    edges[entityNode.getNodeID()][VPNode.getNodeID()].setEdgeType(Node.NODE_TYPE_VERB);
                }

            }
        } else {// len!=1
            Node node0 = new Node();
            Node node_en = new Node();

            if (node.getNodeType() == Node.NODE_TYPE_ENTITY) { // current node is an entity node
                // generate VP node first
                node0 = createNode(Node.NODE_TYPE_VERB);
                ComputeNodeStartEnd(node0, treeNode); // calculate start and end
                edges[node.getNodeID()][node0.getNodeID()].setEdgeType(Edge.TYPE_VERB); // set the edge
                listg.add(node0.getNodeID()); // add it to the result list

                if (node0.getStr().matches("(.*?) ((is|was|are|were) )?also (.*)")) { // the case with 'also' is easily misidentified
                    return listg;
                }

            } else if (node.getNodeType() == Node.NODE_TYPE_ROOT) { // current node is a root node
                node0 = node;
                node_en = nodes[node.getNodeID() + 1]; // get node_en as the entity connected to root
            } else { // current node is description
                node0 = node;
            }


            // further breakdown according to special circumstances
            if (len == 2) { // SQ has two children

                TreeNode treeNode1 = treeNode.children.get(0); // left child
                TreeNode treeNode2 = treeNode.children.get(1); // right child

                if (treeNode1.data.startsWith("VB")) { // VBZ, VBP, VBD, VBG, VBN?

                    if (judgeIfAuxiliary(treeNode1.str)) { // left child is a auxiliary verb
                        if (treeNode2.data.equals("SBAR")) { // subordinate clause detected
                            String sbarString = selectLeaf(treeNode2);
                            if (sbarString.contains(", ")) { // comma-guided subordinate clause of synonym
                                String[] strArr = sbarString.split(",");
                                String origin = strArr[0].trim();
                                String clause = strArr[strArr.length - 1].trim();
                                if (!clause.equals("") && node0.getNodeType() == Node.NODE_TYPE_VERB) {
                                    // create a new NVP node and connect to node
                                    Node nvpNode = createNode(Node.NODE_TYPE_NON_VERB); // VPNode is already generated
                                    nvpNode.setEnd(node0.getEnd());
                                    nvpNode.setStart(node0.getEnd() - clause.split(" ").length);
                                    nvpNode.setStr(clause.replaceAll("(which|what|who|where|when) ", ""));// remove the lead word

                                    // modify the string of original VP
                                    node0.setEnd(nvpNode.getStart() - 1);
                                    node0.setStr(origin);
                                }

                            } else {
                                logger.error("SQ not Cover-3, SQ guided by auxiliary verb:" + question);
                            }
                        } else if (treeNode2.data.trim().equals("NP")) {
                            logger.error("SQ not Cover-2, auxiliary verb + NP:" + question);
                        } else {
                            logger.error("SQ not Cover-2, SQ guided by auxiliary verb:" + question);
                        }
                    } else if (judgeIfBeWord(treeNode1.str)
                            && treeNode.parent.children.get(0).data.trim().equals("WHPP")) { // be verb, and SQ is preceded by WHPP
                        // e.g. "In what nn is xxx"
                        if (treeNode2.data.trim().equals("NP")) { // the right child is NP
                            if (containClause(treeNode2)) {
                                NP(treeNode2, node0); // node0 is either a new entity node, description node, or a root node
                            }
                        } else if (treeNode2.data.trim().equals("VP")) { // passive
                            // continue processing by substituting into SQ
                            List<Integer> listg2 = SQ(treeNode2, node0);
                            if (listg2.size() > 0) {
                                for (int in2 : listg2) {
                                    edges[node0.getNodeID()][in2].edgeType = Edge.TYPE_REFER;
                                }
                            }
                        } else {
                            logger.error("SQ not Cover-3:" + question);
                        }
                    } else if (judgeIfBeWord(treeNode1.str) && ((treeNode2.data.trim().equals("VP")
                            || (treeNode2.data.trim().equals("S") && treeNode2.children.size() == 1 && treeNode2.children.get(0).data.trim().equals("VP "))))) { // passive
                        // e.g., what city is owned by
                        // tree2 is either VP or S, VP is under S
                        // put it to SQ to process recursively
                        List<Integer> listg1 = SQ(treeNode2, node0);
                        if (listg1.size() > 0) {

                            node0.setStart(node0.getStart() - 1); // add be verb to node0
                            for (Integer in1 : listg1) {
                                if (node0.getNodeType() == Node.NODE_TYPE_ENTITY) {
                                    edges[node0.getNodeID()][in1].edgeType = Edge.TYPE_VERB;
                                }
                            }
                        }

                    } else {
                        if (treeNode2.data.startsWith("NP")) { // e.g. What is xxx
                            if (containClause(treeNode2) || node.getNodeType() == Node.NODE_TYPE_ROOT
                                    || edges[0][1].edgeType == Edge.TYPE_NO_EDGE) {
                                NP(treeNode2, node0);
                            }
                        } else if (treeNode2.data.startsWith("PP")) {//e.g. what is of xxx
                            if (containClause(treeNode2)) {
                                PP(treeNode2, node0);
                            }
                        } else if (treeNode2.data.trim().equals("ADJP") || treeNode2.data.trim().equals("ADVP")) { // e.g. what xxx is adj./adv.
                            if (treeNode2.children.size() == 2) {
                                TreeNode treeNode21 = treeNode2.children.get(0);
                                TreeNode treeNode22 = treeNode2.children.get(1);
                                if (treeNode21.data.startsWith("JJ") && treeNode22.data.trim().equals("S")) {
                                    S(treeNode22, node0);
                                } else {
                                    logger.error("SQ not Cover-4:" + question);
                                }
                            }
                        } else if (treeNode2.data.startsWith("VP")) { // e.g. what xxx is -ed by  /has come ...
                            // recursive to SQ
                            List<Integer> listg2 = SQ(treeNode2, node0);
                            if (listg2.size() > 0) {
                                for (int in2 : listg2) {
                                    edges[node0.getNodeID()][in2].edgeType = Edge.TYPE_REFER;
                                }
                            }
                        } else if (treeNode2.data.trim().equals("S")) {//e.g. ?
                            if (treeNode2.children.size() == 2
                                    && treeNode2.children.get(0).data.trim().equals("NP")
                                    && treeNode2.children.get(1).data.trim().equals("VP")) {
                                TreeNode treeNode21 = treeNode2.children.get(0);//NP
                                TreeNode treeNode22 = treeNode2.children.get(1);//VP

                                if (treeNode21.children.size() == 2) { // two nodes are under NP
                                    TreeNode treeNode211 = treeNode21.children.get(0);
                                    TreeNode treeNode212 = treeNode21.children.get(1);
                                    if (treeNode211.data.equals("NP ") && treeNode212.data.equals("NP ")) {

                                        // create a new entity node
                                        Node node1 = new Node();
                                        node1.setNodeID(numNode);
                                        nodes[numNode] = node1;
                                        node1.setNodeType(Node.NODE_TYPE_ENTITY);
                                        numNode++;

                                        // an reference edge
                                        edges[node0.getNodeID()][node1.getNodeID()].edgeType = Edge.TYPE_REFER;
                                        ComputeEdgeStartEnd(node0.getNodeID(), node1.getNodeID(), treeNode22);

                                        // handle it to NP
                                        List<Integer> listg1 = NP(treeNode212, node1);

                                        // NVP edge
                                        edges[node1.getNodeID()][listg1.get(0)].edgeType = Edge.TYPE_NON_VERB;
                                        edges[node0.getNodeID()][node1.getNodeID()].start = nodes[listg1.get(0)].getStart();

                                        // tree2 is handled to SQ
                                        List<Integer> listg2 = SQ(treeNode22, node1);
                                        for (Integer integer : listg2) {
                                            edges[node1.getNodeID()][integer].edgeType = Edge.TYPE_VERB;
                                        }
                                    } else if (treeNode211.data.equals("NP ") && treeNode212.data.equals("PP ")) {
                                        Node node1 = new Node();
                                        node1.setNodeID(numNode);
                                        nodes[node1.getNodeID()] = node1;
                                        node1.setNodeType(2);
                                        numNode++;
                                        edges[node0.getNodeID()][node1.getNodeID()].edgeType = 2;
                                        ComputeEdgeStartEnd(node0.getNodeID(), node1.getNodeID(), treeNode2);

                                        List<Integer> listg1 = NP(treeNode21, node1);
                                        List<Integer> listg2 = SQ(treeNode22, node1);
                                        for (Integer in1 : listg1) {
                                            edges[node1.getNodeID()][in1].edgeType = 4;
                                        }
                                        for (Integer in2 : listg2) {
                                            edges[node1.getNodeID()][in2].edgeType = 3;
                                        }
                                    } else {
                                        Node node1 = new Node();
                                        node1.setNodeID(numNode);
                                        nodes[numNode] = node1;
                                        node1.setNodeType(2);
                                        numNode++;
                                        edges[node0.getNodeID()][node1.getNodeID()].edgeType = 2;
                                        ComputeEdgeStartEnd(node0.getNodeID(), node1.getNodeID(), treeNode2);
                                        List<Integer> listg1 = NP(treeNode21, node1);
                                        edges[node1.getNodeID()][listg1.get(0)].edgeType = 4;
                                        List<Integer> listg2 = SQ(treeNode22, node1);
                                        for (int i = 0; i < listg2.size(); i++) {
                                            edges[node1.getNodeID()][listg2.get(i)].edgeType = 3;
                                        }
                                    }
                                } else {
                                    Node node1 = new Node();
                                    node1.setNodeID(numNode);
                                    nodes[numNode] = node1;
                                    node1.setNodeType(2);
                                    numNode++;
                                    edges[node0.getNodeID()][node1.getNodeID()].edgeType = 2;
                                    ComputeEdgeStartEnd(node0.getNodeID(), node1.getNodeID(), treeNode2);
                                    List<Integer> listg1 = NP(treeNode21, node1);
                                    edges[node1.getNodeID()][listg1.get(0)].edgeType = 4;
                                    List<Integer> listg2 = SQ(treeNode22, node1);
                                    for (int i = 0; i < listg2.size(); i++) {
                                        edges[node1.getNodeID()][listg2.get(i)].edgeType = 3;
                                    }

                                }
                            } else if (treeNode2.children.size() == 1 && treeNode2.children.get(0).data.equals("VP ")) {
                                logger.error("SQ not Cover-5:" + question);
                            } else {
                                logger.error("SQ not Cover-7:" + question);
                            }
                        } else if (treeNode2.data.trim().equals("WHPP")) {// e.g. ?

                            TreeNode treeNode21 = treeNode2.children.get(0); // prepositions
                            TreeNode treeNode22 = treeNode2.children.get(1);
                            String tempString = selectLeaf(treeNode22, "");
                            if (edges[0][1].edgeType == Edge.TYPE_NO_EDGE) {
                                if (tempString.trim().startsWith("how")) { // question of type COUNT
                                    nodes[0].setQuesType(QueryType.COUNT);
                                    String[] tempString1 = selectLeaf(treeNode22, "").toLowerCase().split(" ");
                                    nodes[0].setTrigger(tempString1[1] + " " + tempString1[2]);
                                } else { // question of type COMMON
                                    nodes[0].setQuesType(QueryType.COMMON);
                                    nodes[0].setTrigger(selectLeaf(treeNode2, "").toLowerCase().split(" ")[2]);
                                }
                                int in1 = WH(treeNode22, nodes[1]);
                                edges[0][in1].edgeType = 1;
                                ComputeNodeStartEnd(nodes[0], treeNode22);

                            }
                        } else if (treeNode2.data.trim().equals("WHNP")) { // which book
                            if (edges[0][1].edgeType == Edge.TYPE_NO_EDGE) { // Node0-1 has no edge
                                WH(treeNode2, nodes[1]);
                                ComputeNodeStartEnd(nodes[0], treeNode2);
                                edges[0][1].edgeType = Edge.TYPE_QUEST;
                                nodes[0].setTrigger(selectLeaf(treeNode2, "").trim().split(" ")[0]);
                                nodes[0].setQuesType(QueryType.COMMON);
                            } else { // Node 0-1 has an edge, the new entity needs to be referred
                                // create a new entity node
                                Node node1 = createNode(Node.NODE_TYPE_ENTITY);
                                WH(treeNode2, node1);
                                edges[node0.getNodeID()][node1.getNodeID()].edgeType = Edge.TYPE_REFER;
                                ComputeEdgeStartEnd(node0.getNodeID(), node1.getNodeID(), treeNode2);
                            }

                        } else if (treeNode2.data.trim().equals("SBAR") || treeNode2.data.trim().equals("SBARQ")) { // subordinate clause


                            if (treeNode2.children.size() == 1) {
                                listg = SQ(treeNode2.children.get(0), node0);
                            } else {
                                TreeNode treeNode21 = treeNode2.children.get(0);
                                TreeNode treeNode22 = treeNode2.children.get(1);
                                if (treeNode21.data.equals("WHNP ") && treeNode22.data.equals("SBAR ")) {

                                }
                                // subordinate clause，it refers to a new entity

                                Node node1 = createNode(Node.NODE_TYPE_ENTITY);
                                edges[node0.getNodeID()][node1.getNodeID()].edgeType = Edge.TYPE_REFER;
                                ComputeEdgeStartEnd(node0.getNodeID(), node1.getNodeID(), treeNode2);

                                // handle it to SBAR
                                SBAR(treeNode2, node1, false);

                            }
                        }
                    }
                } else if (treeNode2.data.startsWith("VB")) { // the right child of SQ is VB

                    // give it to SQ to handle recursively
                    SQ(treeNode2, node0);

                    logger.error("SQ not Cover-8:" + question);

                } else if (treeNode1.data.equals("NP ") && treeNode2.data.equals("VP ")) {

                    if (containClause(treeNode1)) { // if there's subordiante clause, process with NP
                        List<Integer> listg1 = NP(treeNode1, node0);
                    } else { // there's not subordinate clause, generate NVP node separately
                        Node newNVPNode = createNode(Node.NODE_TYPE_NON_VERB);
                        ComputeNodeStartEnd(newNVPNode, treeNode1);
                        if (node_en.getNodeType() == Node.NODE_TYPE_ENTITY) {
                            edges[node_en.getNodeID()][newNVPNode.getNodeID()].setEdgeType(Edge.TYPE_NON_VERB);
                        } else {
                            edges[findEntityNodeID(node0)][newNVPNode.getNodeID()].setEdgeType(Edge.TYPE_NON_VERB);
                        }

                        // replace the current NP that appears in another Node
                        if (node0.getStr() != null) {
                            ComputeNodeStartEnd(node0, treeNode2);
                        }

                    }
                    List<Integer> listg2 = SQ(treeNode2, node0);

                } else if ((treeNode1.data.startsWith("TO") || treeNode1.data.startsWith("IN")) && (treeNode2.data.equals("VP "))) {
                    List<Integer> listg1 = SQ(treeNode2, node0);
                    if (listg1.size() > 0) {
                        for (int in1 : listg1) {
                            edges[node0.getNodeID()][in1].edgeType = 2;
                        }
                    }
                } else if (treeNode1.data.equals("MD ") && treeNode2.data.equals("VP ")) {
                    SQ(treeNode2, node0);
                } else if (treeNode1.data.equals("SQ ") && treeNode2.data.equals("SBAR ")) {

                    numNode--; // remove the newly added node

                    List<Integer> listg1 = SQ(treeNode1, node);
                    for (Integer in1 : listg1) {
                        edges[node.getNodeID()][in1].edgeType = Edge.TYPE_VERB;
                    }
                    SBAR(treeNode2, node, false);

                } else if (treeNode1.data.equals("SQ ") && treeNode2.data.equals("VP ")) {

                    numNode--; // remove the newly added node
                    List<Integer> listg1 = SQ(treeNode1, node);
                    for (Integer in1 : listg1) {
                        edges[node.getNodeID()][in1].edgeType = Edge.TYPE_VERB;

                    }
                    List<Integer> listg2 = SQ(treeNode2, node);
                    for (Integer in2 : listg2) {
                        edges[node.getNodeID()][in2].edgeType = Edge.TYPE_VERB;
                    }

                } else {

                    logger.error("SQ not Cover-9:" + question);
                }
            } else if (len == 3) { // there's 3 nodes under SQ
                TreeNode treeNode1 = treeNode.children.get(0);
                TreeNode treeNode2 = treeNode.children.get(1);
                TreeNode treeNode3 = treeNode.children.get(2);
                String ss = selectLeaf(treeNode1, "").toLowerCase().trim() + " ";

                if (treeNode1.data.startsWith("VB") || treeNode1.data.equals("VP ")) {
                    boolean existbe = false;
                    if (WordList.isContainList(WordList.be_form, ss.substring(1))) {
                        existbe = true;
                    }
                    if (treeNode1.data.startsWith("VB") && judgeIfContainAux(ss)) { // inversion
                        if ((treeNode2.data.equals("NP ")) && (treeNode3.data.equals("VP "))) {
                            if (containClause(treeNode2)) {
                                NP(treeNode2, node0);
                            }
                            SQ(treeNode3, node0);
                        } else if ((treeNode2.data.equals("NP ")) && (treeNode3.data.equals("PP "))) {
                            //System.out.println(syntaxTreeText);
                            if (containClause(treeNode2)) {
                                NP(treeNode2, node0);
                            }
                            if (containClause(treeNode2)) {
                                PP(treeNode3, node0);
                            }
                        }

                    } else if (treeNode1.data.startsWith("VB") && (judgeIfContainBeWord(ss))
                            && IsPPOnlyIN(treeNode3) && treeNode2.data.equals("NP ") && treeNode3.data.equals("PP ")) { // inversion, e.g., What country is Mount Everest in?
                        if (containClause(treeNode2)) {
                            NP(treeNode2, node0);
                        }

                    } else if (treeNode1.data.startsWith("VB") && (judgeIfContainBeWord(ss))
                            && treeNode2.data.equals("NP ") && treeNode3.data.equals("VP ")) { // inversion

                        if (containClause(treeNode2)) {
                            NP(treeNode2, node0);
                        }
                        SQ(treeNode3, node0);

                    } else if (treeNode1.data.startsWith("VB") && (judgeIfContainBeWord(ss))
                            && treeNode2.data.equals("NP ") && treeNode3.data.equals("NP ") && sign == 1) {//Yes/no

                        if (containClause(treeNode2)) {
                            NP(treeNode2, node0);
                        }
                        if (containClause(treeNode3)) {
                            NP(treeNode3, node0);
                        }
                        //System.out.println(question);
                        //System.out.println(syntaxTreeText);
                    } else if (treeNode1.data.equals("VP ") && treeNode2.data.equals("CC ") && (treeNode3.data.equals("VP ") || treeNode3.data.equals("S ") || treeNode3.data.equals("SQ "))) {
                        // concatenated conjunctions, e.g., is xx and written by xx
                        if (node.getNodeType() == Node.NODE_TYPE_ENTITY) {
                            listg.clear();
                            numNode--;
                            List<Integer> listg1 = SQ(treeNode1, node);
                            List<Integer> listg2 = SQ(treeNode3, node);
                            if (listg1.size() > 0 && listg2.size() > 0) {
                                listg.addAll(listg1);
                                listg.addAll(listg2);
                            }
                        } else { // the node is not entity, but description

                            // node is replaced by tree1
                            ComputeNodeStartEnd(node, treeNode1);
                            SQ(treeNode1, node);

                            // genereate a new VP node to store tree3
                            Node node1 = createNode(Node.NODE_TYPE_VERB);
                            ComputeNodeStartEnd(node1, treeNode3);
                            // note that it is connected to entity node
                            edges[findEntityNodeID(node)][node1.getNodeID()].setEdgeType(Edge.TYPE_VERB);

                            if (treeNode3.data.equals("S ")) {
                                S(treeNode3, node1);
                            } else {
                                SQ(treeNode3, node1);// tree3 regenerates a node
                            }

                        }
                    } else if (treeNode2.data.equals("NP ") && treeNode3.data.equals("SBAR ")) {// NP + SBAR: subordinate pattern, it needs the reference

                        // create a new entity node, and create an reference edge
                        Node node1 = createNode(Node.NODE_TYPE_ENTITY);
                        edges[node0.getNodeID()][node1.getNodeID()].edgeType = Edge.TYPE_REFER;
                        ComputeEdgeStartEnd(node0.getNodeID(), node1.getNodeID(), treeNode3);
                        edges[node0.getNodeID()][node1.getNodeID()].start = getFirstLeaf(treeNode2).index;

                        List<Integer> listg1 = NP(treeNode2, node1);
                        for (Integer in1 : listg1) {
                            edges[node1.getNodeID()][in1].edgeType = Edge.TYPE_NON_VERB;
                        }
                        SBAR(treeNode3, node1, false);
                    } else if (treeNode2.data.equals("NP ") && treeNode3.data.equals("S ")) {
                        NP(treeNode2, node0);

                        // crate a new entity node
                        Node node1 = createNode(Node.NODE_TYPE_ENTITY);
                        edges[node0.getNodeID()][node1.getNodeID()].edgeType = Edge.TYPE_REFER;
                        ComputeEdgeStartEnd(node0.getNodeID(), node1.getNodeID(), treeNode3);
                        edges[node0.getNodeID()][node1.getNodeID()].start = getFirstLeaf(treeNode2).index;

                        // create a new NVP Node
                        Node node2 = createNode(Node.NODE_TYPE_NON_VERB);
                        edges[node1.getNodeID()][node2.getNodeID()].edgeType = Edge.TYPE_NON_VERB;
                        ComputeNodeStartEnd(node2, treeNode2);
                        S(treeNode3, node1);

                    } else {
                        if (treeNode3.data.equals("NP ")) {
                            NP(treeNode, node0);
                            if (treeNode2.data.equals("NP ")) {
                                logger.error("SQ not Cover-11:" + question);
                            }
                        }
                        if ((treeNode2.data.equals("NP ")) && (treeNode3.data.equals("PP "))) {
                            if (containClause(treeNode2)) {
                                NP(treeNode2, node0);
                            }
                            if (containClause(treeNode3)) {
                                PP(treeNode3, node0);
                            }

                        } else if ((treeNode2.data.equals("NP ") && (treeNode3.data.equals("VP ")))) {

                            if (treeNode3.children.get(0).data.startsWith("VBN")) { // perfect tense
                                if (containClause(treeNode2)) {
                                    NP(treeNode2, node0);
                                }
                                SQ(treeNode3, node0);
                            } else {
                                if (containClause(treeNode2)) {
                                    NP(treeNode2, node0);
                                }
                                SQ(treeNode3, node0);
                            }

                        } else if (treeNode2.data.equals("PP ") && treeNode3.data.equals("PP ")) {

                            if (containClause(treeNode2)) {
                                PP(treeNode2, node0);
                            }
                            if (containClause(treeNode3)) {
                                PP(treeNode3, node0);
                            }

                        } else if (treeNode2.data.equals("NP ") && treeNode3.data.equals("NP ")) {
                            if (containClause(treeNode2)) {
                                NP(treeNode2, node0);
                            }
                            if (containClause(treeNode3)) {
                                NP(treeNode3, node0);
                            }

                        } else if (treeNode2.data.equals("NP ")) { // e.g. is Mount McKinley(NP) located(ADJP)
                            if (containClause(treeNode2)) {
                                NP(treeNode2, node0);
                            }
                            if (treeNode3.data.equals("ADJP ")) {
                            } else if (existbe) {
                            }
                        } else if (treeNode3.data.equals("PP ")) { // e.g. established earlier than 1400
                            if (containClause(treeNode3)) {
                                PP(treeNode3, node0);
                            }
                            if (existbe) {
                                if (treeNode2.data.equals("ADJP ") || treeNode2.data.equals("ADVP ")) {

                                } else if (treeNode2.data.equals("VP ")) {

                                }
                            }
                        }
                    }
                } else if (treeNode1.data.startsWith("MD")) {
                    if (treeNode2.data.equals("NP ") && treeNode3.data.equals("VP ")) { // e.g. can you pay...
                        if (containClause(treeNode2)) {
                            List<Integer> listg1 = NP(treeNode2, node0);
                        }
                        SQ(treeNode3, node0);
                    }
                } else if (treeNode1.data.startsWith("VP") && treeNode2.data.equals("NP ") && treeNode3.data.equals("NP ")) {

                } else if (treeNode1.data.equals("NP ")) {
                    if (containClause(treeNode1)) {
                        NP(treeNode1, node0);
                    }
                    if (treeNode2.data.equals("VP ")) {
                        SQ(treeNode2, node0);
                    }
                    if (treeNode3.data.equals("PP ") && containClause(treeNode3)) {
                        PP(treeNode3, node0);
                    }
                }
            } else {
                logger.error("SQ not Cover-10:" + question);
            }
        }
        return listg;
    }

    /**
     * process the 'PP' phrase, e.g. `... of the xx that ..`, referred entity will be generated
     *
     * @param treeNode the syntax treeNode node tagged 'PP'
     * @param node     the EDG node to be attached
     * @return the IDs of the nodes generated
     */
    private List<Integer> PP(TreeNode treeNode, Node node) { // call it when new nodes are generated
        List<Integer> listg = new ArrayList<>();
        if (treeNode.children.size() != 2 && !treeNode.children.get(0).data.equals("ADVP ")) {
            //System.out.println("Error in PP");
            return listg;
        }
        TreeNode temp = treeNode.children.get(treeNode.children.size() - 1);// PP the most right node
        boolean isRooten = false;
        if (node.getNodeType() == 1) {
            isRooten = true;
        }
        if (temp.data.startsWith("NP")) {

            if (temp.getChildren().size() == 2 &&
                    temp.getFirstChild().getData().trim().equals("NP")
                    && temp.getLastChild().getData().trim().equals("PP")
                    && containClause(temp.getLastChild())) {
                // PP is nested in NP, and NP of PP contains subordinate clauses, then the new entity is needed
                // e.g. the total number of the cast number of the television shows whose actress is Joey Mclntyre
                Node entityNode = createNode(Node.NODE_TYPE_ENTITY);
                ComputeEdgeStartEnd(node.getNodeID(), entityNode.getNodeID(), temp);

                Node desNode = createNode(Node.NODE_TYPE_NON_VERB);
                edges[entityNode.getNodeID()][desNode.getNodeID()].setEdgeType(Edge.TYPE_NON_VERB);
                ComputeNodeStartEnd(desNode, temp);

                NP(temp, desNode);

            } else {
                List<Integer> listg1 = NP(temp, node);
                if (listg1.size() > 0) {
                    listg.add(listg1.get(0)); // without preposition
                }
            }

        } else if (temp.data.startsWith("SBAR")) {
            Node node_en = new Node();
            if (isRooten) {
                node_en = nodes[node.getNodeID() + 1];
            } else {
                node_en.setNodeID(numNode);
                nodes[numNode] = node_en;
                node_en.setNodeType(2);
                numNode++;
            }
            edges[node.getNodeID()][node_en.getNodeID()].edgeType = 2;
            ComputeEdgeStartEnd(node.getNodeID(), node_en.getNodeID(), temp);
            List<Integer> listg1 = SBAR(temp, node_en, isRooten);
            for (Integer in1 : listg1) {
                listg.add(in1);
            }
        } else if (temp.data.equals("PP ")) {
            List<Integer> listg1 = PP(temp, node);
            int l = selectLeaf(treeNode.children.get(0), "").split(" ").length - 1;
            for (int in1 : listg1) {
                nodes[in1].setStart(nodes[in1].getStart() - l);
                listg.add(in1);
            }
        }
        return listg;
    }

    /**
     * process the 'NP' phrase
     *
     * @param treeNode the syntax treeNode node tagged 'NP'
     * @param node     current EDG node (can be root/entity/description node)
     * @return the IDs of the nodes generated
     */
    private List<Integer> NP(TreeNode treeNode, Node node) {
        List<Integer> nodeIdxs = new ArrayList<>();

        int len = treeNode.children.size(); // the number of children of current node
        Node node0 = null; // temp node for the generation of description

        boolean isRooten = false; // if it's a root entity
        if (node.getNodeType() == Node.NODE_TYPE_ROOT) { // if current node is a root node, the additional entity should be generated
            isRooten = true;
        }

        Node entityNode = new Node();
        if (node.getNodeType() == Node.NODE_TYPE_ENTITY) {
            // if this layer generates a new non-ver description node, nodes generated afterwards are connected to the current node, otherwise it is connected to the upper-level verb/non-verb nodes
            // create a new NVP node
            node0 = createNode(Node.NODE_TYPE_NON_VERB);

            ComputeNodeStartEnd(node0, treeNode);
            edges[node.getNodeID()][node0.getNodeID()].edgeType = Edge.TYPE_NON_VERB;

            nodeIdxs.add(node0.getNodeID()); // nodeID of newly generated nodes
        } else if (isRooten) { // root Entity
            node0 = node;
            entityNode = nodes[node.getNodeID() + 1]; // entity node after root


        } else { // it's neither an entity nor root, it's description node
            node0 = node;
        }

        if (containClause(treeNode)) { // current tree contains subordinate clauses
            if (len > 1 && treeNode.getFirstChild().data.trim().equals("NP")) { // the most left child is NP
                // current node has at least two children, and the most left one is NP
                TreeNode NPNode = treeNode.children.get(0); // NPNode
                TreeNode treeNode2 = treeNode.children.get(1); // otherNode
                if (len > 2) {// current node has at least three children
                    TreeNode treeNode3 = treeNode.children.get(2);
                    if (treeNode2.data.trim().equals("PP") && treeNode3.data.trim().equals("SBAR")) { // NP+PP+SBAR

                        if (node.getNodeType() != Node.NODE_TYPE_ROOT) { // current node is not root, but an existing entity
                            // create a new entity node and add it to nodes
                            entityNode = createNode(Node.NODE_TYPE_ENTITY);
                            // record an reference edge
                            edges[node0.getNodeID()][entityNode.getNodeID()].edgeType = Edge.TYPE_REFER;
                            // calculate the start and end of reference edge
                            ComputeEdgeStartEnd(node0.getNodeID(), entityNode.getNodeID(), treeNode);
                        } else {
                            logger.error("NP not Cover-1:" + question);
                        }

                        // recursive
                        List<Integer> desNodeIdxs = NP(NPNode, entityNode);
                        int nvpNodeIdx = desNodeIdxs.get(0);// nodeID of NVP node
                        edges[entityNode.getNodeID()][nvpNodeIdx].edgeType = Edge.TYPE_NON_VERB; // nvp edge

                        // SBAR Node
                        List<Integer> desNodeIdxs2 = SBAR(treeNode3, entityNode, isRooten);
                        nodeIdxs.addAll(desNodeIdxs2);
                        if (containClause(treeNode2)) {// if tree2 has subordinate clauses
                            PP(treeNode2, node0);
                        }

                        Node node1 = new Node();
                        ComputeNodeStartEnd(node1, treeNode2);
                        nodes[nvpNodeIdx].setEnd(node1.getEnd());

                    } else if (NPNode.data.trim().equals("NP") && treeNode2.data.trim().equals(",") && treeNode3.data.trim().equals("NP")) {
                        if (node.getNodeType() == Node.NODE_TYPE_ROOT) {  // connected directly to root
                            logger.error("NP not Cover-2:" + question);
                        } else { // not connected directly to root
                            if (containClause(NPNode)) {
                                logger.error("NP not Cover-3:" + question);
                            } else {
                                List<Integer> listg1 = NP(treeNode3, node0); // recursive

                            }
                        }
                    } else if (treeNode3.data.startsWith("SBAR")) {
                        if (node.getNodeType() != Node.NODE_TYPE_ROOT) {
                            // current node is not root

                            entityNode.setNodeID(numNode);
                            nodes[numNode] = entityNode;
                            numNode++;
                            entityNode.setNodeType(Node.NODE_TYPE_ENTITY); // new entity

                            // create a new reference edge
                            edges[node0.getNodeID()][entityNode.getNodeID()].edgeType = Edge.TYPE_REFER;

                            ComputeEdgeStartEnd(node0.getNodeID(), entityNode.getNodeID(), treeNode);
                        } else {
                            logger.error("NP not Cover-4:" + question);

                        }

                        nodeIdxs.addAll(SBAR(treeNode3, entityNode, isRooten));

                        List<Integer> listg1;
                        if (NPNode.data.trim().equals("NP") && treeNode2.data.trim().equals(",")) {
                            listg1 = NP(NPNode, entityNode);
                        } else {
                            listg1 = NP(NPNode, entityNode);
                            for (Integer in1 : listg1) {
                                nodes[in1].setEnd(getLastLeaf(treeNode2).index + 1);
                            }
                        }
                        if (isRooten) { // the node is root
                            nodeIdxs.addAll(listg1);
                        }
                    } else if (treeNode2.data.trim().startsWith("SBAR")) {
                        // create a new entity, and set the reference edge
                        Node referEntity = createNode(Node.NODE_TYPE_ENTITY);
                        edges[node0.getNodeID()][referEntity.getNodeID()].setEdgeType(Edge.TYPE_REFER);
                        ComputeEdgeStartEnd(node0.getNodeID(), referEntity.getNodeID(), treeNode);

                        List<Integer> desNodeIdxs = NP(NPNode, referEntity);
                        int nvpNodeIdx = desNodeIdxs.get(0); // nodeID of NVP node
                        edges[referEntity.getNodeID()][nvpNodeIdx].setEdgeType(Edge.TYPE_NON_VERB); // nvp edge

                        // create a new VPNode and input the part after SBAR
                        Node vpDesNode = createNode(Node.NODE_TYPE_VERB);
                        edges[referEntity.getNodeID()][vpDesNode.getNodeID()].setEdgeType(Edge.TYPE_VERB); // VP edge
                        vpDesNode.setStart(getFirstLeaf(treeNode2).index);
                        vpDesNode.setEnd(getLastLeaf(treeNode3).index);
                        vpDesNode.setStr(selectLeaf(treeNode2).trim() + " " + selectLeaf(treeNode3).trim());
                    } else {
                        logger.error("NP not Cover-5:" + question);
                    }
                } else {
                    // current node has exactly two children, NP + XX
                    if (containClause(NPNode)) {
                        logger.error("NP not Cover-6:" + question);
                    }

                    if (treeNode2.data.trim().equals("PP")) { // NP + PP
                        // create a new nvp Node
                        Node nvpDesNode = new Node();
                        if (node.getNodeType() == Node.NODE_TYPE_ROOT) { // current node is root
                            nvpDesNode.setNodeID(numNode);
                            nodes[numNode] = nvpDesNode;
                            numNode++;
                            nvpDesNode.setNodeType(Node.NODE_TYPE_NON_VERB); // the node type is nvp
                            ComputeNodeStartEnd(nvpDesNode, treeNode);
                            nodeIdxs.add(nvpDesNode.getNodeID());
                            edges[entityNode.getNodeID()][nvpDesNode.getNodeID()].edgeType = Edge.TYPE_NON_VERB;// the edge type is nvp
                        } else {  // current node is not root
                            nvpDesNode = node0; // get NVP node as description node
                        }

                        // put tree2,nvpDesNode to PP
                        List<Integer> nodeIdxsAfterPP = PP(treeNode2, nvpDesNode);
                        if (nodeIdxsAfterPP.size() > 1) { // the number of generated nodes >= 2
                            logger.error("NP not Cover-7:" + question);
                        }
                        if (nodeIdxsAfterPP.size() > 0) { // exactly one node is generated
                            int in2 = nodeIdxsAfterPP.get(0);
                            if (nodes[in2].getNodeType() == Node.NODE_TYPE_ENTITY) { // a new entity node
                                edges[nvpDesNode.getNodeID()][in2].edgeType = Edge.TYPE_REFER; // reference
                                ComputeEdgeStartEnd(nvpDesNode.getNodeID(), in2, treeNode2.children.get(1));
                                if (node.getNodeType() != 2) {
                                    nodeIdxs.add(in2);
                                }
                            } else { // the new node is not entity node
                                logger.error("NP not Cover-8:" + question);
                            }
                        }
                    } else if (treeNode2.data.startsWith("SBAR") || treeNode2.data.trim().equals("VP")) { // NP + SBAR / NP+SBARQ / NP+VP

                        if (treeNode2.data.trim().equals("VP") && treeNode2.getChildren().size() == 1) {
                            // VP of (NP + VP) is one word, no addition entity
                            //e.g. has A written
                            return nodeIdxs;
                        }

                        if (treeNode2.data.trim().equals("SBAR")
                                && selectLeaf(treeNode2).trim().toLowerCase().startsWith("when")
                                && KB == KBEnum.Freebase) {
                            return nodeIdxs;
                        }

                        if (node.getNodeType() != Node.NODE_TYPE_ROOT) { // current node is root
                            entityNode.setNodeID(numNode);
                            nodes[numNode] = entityNode;
                            numNode++;
                            entityNode.setNodeType(Node.NODE_TYPE_ENTITY); // new entity
                            edges[node0.getNodeID()][entityNode.getNodeID()].edgeType = Edge.TYPE_REFER; // reference edge
                            ComputeEdgeStartEnd(node0.getNodeID(), entityNode.getNodeID(), treeNode);
                        } else {
                            logger.error("NP not Cover-9:" + question);
                        }


                        // process NP node at first
                        List<Integer> nvpDesNode = NP(NPNode, entityNode);
                        int in1 = nvpDesNode.get(0); //NVP Node
                        edges[entityNode.getNodeID()][in1].edgeType = Edge.TYPE_NON_VERB; //NVP edge

                        if (treeNode2.data.startsWith("SBAR")) { // NP+SBAR || NP+SBARQ
                            nodeIdxs.addAll(SBAR(treeNode2, entityNode, isRooten));

                        } else { // NP + VP
                            List<Integer> listg2 = SQ(treeNode2, entityNode);
                            for (int in2 : listg2) { // verb node
                                edges[entityNode.getNodeID()][in2].edgeType = Edge.TYPE_VERB;

                                if (node.getNodeType() <= Node.NODE_TYPE_ENTITY) { // current node is root or entity
                                    nodeIdxs.add(in2);
                                }
                            }
                            if (node.getNodeType() > Node.NODE_TYPE_ENTITY) { // current node is description Node
                                nodeIdxs.add(entityNode.getNodeID());
                            }
                        }
                    } else {
                        logger.error("NP not Cover-10:" + question);
                    }
                }
            } else if (len == 3 && treeNode.children.get(2).data.trim().equals("SBAR")) {
                // XX + XX + SBAR
                TreeNode treeNode1 = treeNode.children.get(0);
                TreeNode treeNode2 = treeNode.children.get(1);
                TreeNode SBARNode = treeNode.children.get(2);
                if (node.getNodeType() != Node.NODE_TYPE_ROOT) {
                    // create a new entity node and reference edge
                    entityNode.setNodeID(numNode);
                    nodes[numNode] = entityNode;
                    numNode++;
                    entityNode.setNodeType(Node.NODE_TYPE_ENTITY); // new entity
                    edges[node0.getNodeID()][entityNode.getNodeID()].edgeType = Edge.TYPE_REFER;
                    ComputeEdgeStartEnd(node0.getNodeID(), entityNode.getNodeID(), treeNode);
                } else {
                    logger.error("NP not Cover-11:" + question);
                }
                nodeIdxs.addAll(SBAR(SBARNode, entityNode, isRooten));

                List<Integer> listg1 = NP(treeNode1, entityNode);
                for (Integer in1 : listg1) {
                    nodes[in1].setEnd(getLastLeaf(treeNode2).index + 1);
                }
                if (isRooten) { // current node is root
                    nodeIdxs.addAll(listg1);
                }
            } else { // other cases
                for (TreeNode temp : treeNode.children) {
                    if (temp.data.trim().equals("VP") || temp.data.trim().equals("S") || temp.data.trim().equals("SQ")) {
                        SQ(temp, node0);
                    } else if (temp.data.trim().equals("PP")) {
                        PP(temp, node0);
                    } else {
                        NP(temp, node0);
                    }
                }
            }
        } else { // current tree has no subordinate clauses

            if (node.getNodeType() == Node.NODE_TYPE_ROOT) { // current node is root
                // create a new NVP description node
                Node node1 = createNode(Node.NODE_TYPE_NON_VERB);
                ComputeNodeStartEnd(node1, treeNode);
                edges[entityNode.getNodeID()][node1.getNodeID()].edgeType = Edge.TYPE_NON_VERB;
                nodeIdxs.add(node1.getNodeID());
            }
        }

        // other cases
        if (treeNode.children.size() == 2) {        // current node has two children
            TreeNode treeNode2 = treeNode.children.get(1);
            if (treeNode2.data.trim().equals("WHPP")) { // Of which ...
                if (edges[0][1].edgeType == Edge.TYPE_NO_EDGE) { // no edge from Node0 to Node1
                    WH(treeNode2.children.get(1), nodes[1]);
                    edges[0][1].edgeType = Edge.TYPE_QUEST;
                    ComputeNodeStartEnd(nodes[0], treeNode2);
                    nodes[0].setQuesType(QueryType.COMMON);
                    nodes[0].setTrigger(selectLeaf(treeNode2.children.get(1).children.get(0), "")); // get trigger, e.g., 'which' in 'of (which)'
                } else { // there's edge from Node0 to Node1
                    // create a new entity node
                    Node node1 = new Node();
                    node1.setNodeID(numNode);
                    nodes[node1.getNodeID()] = node1;
                    numNode++;
                    node1.setNodeType(Node.NODE_TYPE_ENTITY);
                    WH(treeNode2.children.get(1), node1);
                    edges[node0.getNodeID()][node1.getNodeID()].edgeType = Edge.TYPE_QUEST;
                    ComputeEdgeStartEnd(node0.getNodeID(), node1.getNodeID(), treeNode2);
                }
            }

        }

        return nodeIdxs;
    }

    /**
     * process the 'SBAR' clause
     *
     * @param treeNode the syntax treeNode node tagged 'SBAR'
     * @param node_en  entity node
     * @param isRooten whether current node is entity0
     * @return the IDs of the nodes generated
     */
    private List<Integer> SBAR(TreeNode treeNode, Node node_en, boolean isRooten) {
        List<Integer> listg = new ArrayList<>();  // the result node list
        List<Integer> listg2 = new ArrayList<>(); // results of recursive functions
        if (treeNode.children.size() == 2) {      // SBAR has two children
            String substr = selectLeaf(treeNode.children.get(0), "").toLowerCase().trim(); // string of the node where the introductory word is located
            if (substr.equals("that") || substr.equals("which")) {
                listg2 = S(treeNode.children.get(1), node_en); // S->VP
                // omitting relational words
            } else if (substr.equals("who") || substr.equals("whom") || substr.equals("when") || substr.equals("where")) {
                listg2 = S(treeNode.children.get(1), node_en); // S->VP

                for (Integer in2 : listg2) { // add information
                    nodes[in2].setStr(substr + " " + nodes[in2].getStr());
                }
            } else if (substr.startsWith("whose")) {
                listg2 = S(treeNode.children.get(1), node_en);
                String rel = substr.trim();

                // put relation after 'whose' to the following description
                if (!listg2.isEmpty()) {
                    Node curNode = nodes[listg2.get(0)];
                    String s = curNode.getStr().trim();
                    s = rel + " " + s;
                    curNode.setStr(s);

                }
                for (Integer in2 : listg2) {
                    edges[node_en.getNodeID()][in2].edgeType = Edge.TYPE_VERB;
                    edges[node_en.getNodeID()][in2].info = substr; // info to edge
                }

            } else if (treeNode.getFirstChild().data.trim().equals("WHPP")) {
                if (isRooten) {
                    TreeNode treeNode1 = treeNode.children.get(0).children.get(0);
                    TreeNode treeNode2 = treeNode.children.get(0).children.get(1);
                    nodes[0].setTrigger(getFirstLeaf(treeNode2).str);
                    edges[0][node_en.getNodeID()].info = treeNode1.str;
                    int in1 = WH(treeNode2, node_en);
                    listg2 = SQ(treeNode.children.get(1), nodes[in1]);
                    for (Integer in3 : listg2) {
                        edges[in1][in3].edgeType = 3;
                        edges[in1][in3].info = selectLeaf(treeNode1, "").substring(1).toLowerCase();
                    }
                } else {
                    TreeNode treeNode2 = treeNode.children.get(1); // S
                    if (treeNode2.children.size() == 1) {
                        listg2 = S(treeNode2, node_en); // S -> VP
                        for (Integer in2 : listg2) {
                            edges[node_en.getNodeID()][in2].info = substr;
                        }
                    } else {
                        // new description node
                        Node desNode = createNode(Node.NODE_TYPE_VERB);
                        StringBuilder desStr = new StringBuilder();
                        for (TreeNode tmpTreeNode : treeNode2.children) {
                            desStr.append(" ").append(selectLeaf(tmpTreeNode).trim());
                        }
                        desNode.setStr(desStr.toString().trim());
                        desNode.setStart(TreeNode.getFirstLeaf(treeNode2).index);
                        desNode.setEnd(TreeNode.getLastLeaf(treeNode2).index);
                        edges[node_en.getNodeID()][desNode.getNodeID()].setEdgeType(Edge.TYPE_VERB);
                    }
                }

            } else if (substr.startsWith("which ") || substr.startsWith("what ")) {
                String npstr = "";
                TreeNode treeNode1 = treeNode.children.get(0);
                for (int i = 1; i < treeNode1.children.size(); i++) {
                    npstr += selectLeaf(treeNode1.children.get(i), "");
                }
                Node node1 = new Node();
                node1.setNodeID(numNode);
                nodes[numNode] = node1;
                node1.setNodeType(4);
                numNode++;
                ComputeNodeStartEnd(node1, treeNode1);
                node1.setStart(node1.getStart() + 1);
                edges[node_en.getNodeID()][node1.getNodeID()].edgeType = 4;

                listg2 = S(treeNode.children.get(1), node_en); // S->VP

            } else {
                listg2 = S(treeNode.children.get(1), node_en); // S->VP

            }
            if (isRooten) {
                for (Integer in2 : listg2) {
                    edges[node_en.getNodeID()][in2].edgeType = 3;
                }
                nodes[0].setQuesType(QueryType.COMMON);
                if (nodes[0].getTrigger() == null) {
                    nodes[0].setTrigger(substr.split(" ")[0]);
                }
            } else if (node_en.getNodeID() == 1) {
                for (Integer in2 : listg2) {
                    edges[node_en.getNodeID()][in2].edgeType = 3;
                }
            } else {
                for (Integer in2 : listg2) {
                    edges[node_en.getNodeID()][in2].edgeType = 2;
                }
            }
        } else { // SBAR has only one child
            TreeNode temptree = treeNode.children.get(0);
            if (temptree.children.size() == 1 && temptree.children.get(0).data.equals("VP ")) {
                listg2 = SQ(temptree.children.get(0), node_en);
            } else if (temptree.data.equals("S ")) {
                listg2 = S(temptree, node_en);
            } else {// other condition, generate VP node
                Node desNode = createNode(Node.NODE_TYPE_VERB);
                ComputeNodeStartEnd(desNode, temptree);
                edges[node_en.getNodeID()][desNode.getNodeID()].setEdgeType(Edge.TYPE_VERB);
            }

        }

        if (listg2.size() > 1) {
            // System.out.println("--2");
        }
        for (int in2 : listg2) { //verb node
            edges[node_en.getNodeID()][in2].edgeType = 3;

            if (isRooten) {
                listg.add(in2);
            }
        }
        if (!isRooten) {
            listg.add(node_en.getNodeID());
        }
        return listg;
    }

    /**
     * process the 'WHNP'/'WHADJP'/'WHAVP' phrase
     *
     * @param treeNode the syntax treeNode node tagged 'WHNP'/'WHADJP'/'WHAVP'
     * @param node     entity node
     * @return the nodeID of node
     */
    private int WH(TreeNode treeNode, Node node) {//common
        if (treeNode.data.trim().equals("WHNP")) {
            int len = treeNode.children.size();
            if (len == 1) {
                // WHNP has only one child
                // e.g., (WHNP (WDT which))
                TreeNode treeNode1 = treeNode.children.get(0);
                if (!(treeNode1.data.trim().startsWith("WP") || treeNode1.data.trim().equals("WDT"))) {

                    String triggerWord = getWHWord(selectLeaf(treeNode1));
                    Node desNode = createNode(Node.NODE_TYPE_VERB);
                    desNode.setStr(selectLeaf(treeNode1).replaceAll("(?i)" + triggerWord, ""));
                    if (triggerWord != null) {
                        desNode.setStart(triggerWord.split(" ").length);
                        desNode.setEnd(taggedQuestion.split(" ").length);
                    }
                    edges[node.getNodeID()][desNode.getNodeID()].setEdgeType(Edge.TYPE_VERB);
                }
            } else { // WHNP has two nodes or more
                TreeNode treeNode1 = treeNode.children.get(0);
                TreeNode treeNode2 = treeNode.children.get(1);
                if (treeNode1.data.trim().equals("WDT")) { // question words

                    // create a new NVP Node
                    Node node1 = createNode(Node.NODE_TYPE_NON_VERB);

                    String npstr = "";
                    for (int i = 1; i < treeNode.children.size(); i++) { // attach the string of all following nodes
                        npstr += selectLeaf(treeNode.children.get(i), "");
                    }

                    edges[node.getNodeID()][node1.getNodeID()].edgeType = Edge.TYPE_NON_VERB; // nvp edge
                    ComputeNodeStartEnd(node1, treeNode2);
                    if (!npstr.equals("")) { // reset the end
                        node1.setEnd(node1.getStart() + npstr.trim().split(" ").length);
                        node1.setStr(npstr); // assign the content after 'what' to node.str
                    }
                } else if (treeNode1.data.equals("WHNP ")) { // WHNP is nested in WHNP
                    int in1;
                    if (treeNode1.children.size() == 1) { // the nested WHNP has only one node
                        TreeNode firstNode = treeNode1.children.get(0);
                        String triggerTag = firstNode.data.trim();
                        if (!(triggerTag.startsWith("WP") || triggerTag.equals("WDT"))) {
                            logger.error("WHNP nesting not cover:" + question);
                        }
                        if (treeNode2.data.trim().equals("NP") || treeNode2.data.trim().startsWith("NN")) {
                            NP(treeNode2, node);
                        } else if (treeNode2.data.trim().equals("PP")) {
                            PP(treeNode2, node);
                        }

                    } else if (treeNode1.children.size() == 2) { // WHNP has two children

                        List<Integer> listg1 = NP(treeNode1.children.get(1), node);
                        if (listg1.size() > 1) {
                            logger.error("WHNP not Cover-2:" + question);
                        }
                        in1 = listg1.get(0); // nodeID returned from NP
                        edges[node.getNodeID()][in1].edgeType = Edge.TYPE_NON_VERB;

                        if (treeNode2.data.equals("PP ")) {  // WHNP + PP
                            if (containClause(treeNode2)) { // tree2 has suboridinate clause
                                List<Integer> listg2 = PP(treeNode2, nodes[in1]);
                                for (int in2 : listg2) {
                                    edges[in1][in2].edgeType = Edge.TYPE_REFER;
                                }
                            }
                            String ppString = selectLeaf(treeNode2, "").trim();
                            String[] t = ppString.split(" "); // the length of PP
                            nodes[in1].setEnd(nodes[in1].getEnd() + t.length);
                            nodes[in1].setStr(nodes[in1].getStr().trim() + " " + ppString);

                        } else if (treeNode2.data.trim().equals("SBAR")) {
                            List<Integer> sbarNodes = SBAR(treeNode2, node, true);
                            for (Integer nodeId : sbarNodes) {
                                edges[node.getNodeID()][nodeId].setEdgeType(Edge.TYPE_VERB);
                            }
                        } else {
                            logger.error("WHNP not Cover-3:" + question);
                        }

                    } else { // WHNP has 3 nodes or more
                        if (treeNode1.children.size() == 3) { //
                            TreeNode treeNode11 = treeNode1.children.get(0);
                            TreeNode treeNode12 = treeNode1.children.get(1);
                            TreeNode treeNode13 = treeNode1.children.get(2);
                            if (treeNode11.data.equals("WDT ") && treeNode12.data.equals("JJ ") && (treeNode13.data.equals("NN ") || treeNode13.data.equals("NP "))) {

                                // create a new NVP node
                                Node node1 = new Node();
                                node1.setNodeID(numNode);
                                nodes[numNode] = node1;
                                numNode++;
                                node1.setNodeType(Node.NODE_TYPE_NON_VERB);
                                node1.setStart(TreeNode.getFirstLeaf(treeNode12).index); // set start as the start of tree12
                                node1.setEnd(getLastLeaf(treeNode13).index + 1); // set end as the end of tree13 + 1
                                edges[node.getNodeID()][node1.getNodeID()].edgeType = Edge.TYPE_NON_VERB;

                                if (treeNode13.data.equals("NP ")) { // tree13 is NP
                                    NP(treeNode13, node);
                                }
                            } else if (treeNode11.data.equals("WDT ") && treeNode12.data.equals("NN ") && treeNode13.data.equals("POS ")) {
                                // possessive, e.g., which xxx 's
                                Node node1 = new Node();
                                node1.setNodeID(numNode);
                                nodes[numNode] = node1;
                                numNode++;
                                node1.setNodeType(Node.NODE_TYPE_NON_VERB);
                                node1.setStart(treeNode12.index);
                                node1.setEnd(getLastLeaf(treeNode).index + 1);
                                edges[node.getNodeID()][node1.getNodeID()].edgeType = Edge.TYPE_NON_VERB;
                            }
                        } else if (treeNode2.data.equals("PP ")) { // tree2 is PP
                            List<Integer> listg2 = PP(treeNode2, node);
                            for (int in2 : listg2) {
                                edges[node.getNodeID()][in2].edgeType = Node.NODE_TYPE_NON_VERB;
                                String[] t = selectLeaf(treeNode2.children.get(0), "").trim().split(" ");
                                nodes[in2].setStart(nodes[in2].getStart() - t.length); // set start
                            }
                        } else if (treeNode2.data.trim().equals("NP")) {
                            NP(treeNode2, node);
                        } else {
                            logger.error("WHNP not Cover-1:" + question);
                        }
                    }

                } else if (treeNode1.data.equals("WHADJP ") || treeNode1.data.equals("WP ")) { // e.g., how hot
                    // create a new NVP node
                    Node node1 = new Node();
                    node1.setNodeID(numNode);
                    nodes[numNode] = node1;
                    numNode++;
                    node1.setNodeType(Node.NODE_TYPE_NON_VERB);

                    String npstr = "";
                    for (int i = 1; i < treeNode.children.size(); i++) {
                        npstr += selectLeaf(treeNode.children.get(i), "");
                    }

                    edges[node.getNodeID()][node1.getNodeID()].edgeType = 4;
                    ComputeNodeStartEnd(node1, treeNode2);
                    if (!npstr.equals("")) {
                        node1.setEnd(node1.getStart() + npstr.trim().split(" ").length);
                        node1.setStr(npstr);
                    }
                } else {
                    logger.error("WHNP not Cover-4:" + question);
                }
            }
        } else if (treeNode.data.equals("WHADVP ")) { // when, where, etc. they can be skipped directly
            if (treeNode.children.size() != 1) {
                logger.error("WHADVP not Cover-1:" + question);
            }
        } else {
            logger.error("WH not Cover-1:" + question);

        }
        return node.getNodeID();
    }

    /**
     * determine whether the questions is a special sentence, handle if it is
     *
     * @return true= special sentence, handle it；false= not special sentence, do not handle it right now
     */
    private boolean JudgeSpecialSent() {


        Matcher matcher3 = Pattern.compile("(?i)((.*) (when|during) (.*))").matcher(question.trim());
        if (KB == KBEnum.Freebase && matcher3.matches()) {
            System.out.println("[DEBUG] SpecialSent Matcher3 Matching:" + question);

            String former = matcher3.group(2);
            String latter = matcher3.group(3) + " " + matcher3.group(4);

            String trigger = former.split(" ")[0];
            former = former.replace(trigger, "").trim();

            // create root Node
            Node rootNode = createNode(Node.NODE_TYPE_ROOT);
            rootNode.setTrigger(trigger);
            rootNode.setQuesType(QueryType.COMMON);
            // crate a new entity node, and connect root to entity node
            Node entityNode = createNode(Node.NODE_TYPE_ENTITY);
            edges[rootNode.getNodeID()][entityNode.getNodeID()].setEdgeType(Edge.TYPE_QUEST);

            Node desNode1 = createNode(Node.NODE_TYPE_VERB);
            desNode1.setStr(former);
            edges[entityNode.getNodeID()][desNode1.getNodeID()].setEdgeType(Edge.TYPE_VERB);

            Node desNode2 = createNode(Node.NODE_TYPE_VERB);
            desNode2.setStr(latter);
            edges[entityNode.getNodeID()][desNode2.getNodeID()].setEdgeType(Edge.TYPE_VERB);

            return true;
        }

        //Which band's former member are Kevin Jonas and Joe Jonas?
        Matcher matcher2 = Pattern.compile("(?i)((which|what|whose) (.*)('s|s') ((.*) (is|are|was|were) (.*)))").matcher(question.trim());
        if (matcher2.matches()) {
            System.out.println("[DEBUG] SpecialSent Matcher2 Matching:" + question);
            if (matcher2.group(3).matches("(.*) (whose|who|which|where|when) (.*)")) {// subordinate clause
                return false;
            }

            // a root Node
            Node rootNode = createNode(Node.NODE_TYPE_ROOT);
            rootNode.setTrigger(matcher2.group(2).toLowerCase());
            rootNode.setQuesType(QueryType.COMMON);

            // new entity node, and connect root and entity node
            Node entityNode = createNode(Node.NODE_TYPE_ENTITY);
            edges[rootNode.getNodeID()][entityNode.getNodeID()].setEdgeType(Edge.TYPE_QUEST);

            if (rootNode.getTrigger().equals("whose")) { // process the special case of 'whose'
                //whose network's parent organisation is Comcast

                // new desNode1, network is #entity1
                Node desNode1 = createNode(Node.NODE_TYPE_NON_VERB);
                desNode1.setStr(matcher2.group(3).trim() + " is #entity1");
                desNode1.setStart(1);
                desNode1.setEnd(question.split(" ").length);
                edges[entityNode.getNodeID()][desNode1.getNodeID()].setEdgeType(Edge.TYPE_NON_VERB);

                // a new entity node as referred entity of descriptionNode1
                Node entityNode1 = createNode(Node.NODE_TYPE_ENTITY);
                edges[desNode1.getNodeID()][entityNode1.getNodeID()].setEdgeType(Edge.TYPE_REFER);
                edges[desNode1.getNodeID()][entityNode1.getNodeID()].setStart(desNode1.getEnd() + 1);
                edges[desNode1.getNodeID()][entityNode1.getNodeID()].setEnd(question.split(" ").length);


                // create a desNode2, as a modifier of entityNode1
                Node desNode2 = createNode(Node.NODE_TYPE_VERB);
                desNode2.setStr(matcher2.group(5)
                        .replace("?", "").
                                replace(".", "").
                                replace("!", "").trim());
                desNode2.setStart(desNode1.getEnd() + 1);
                desNode2.setEnd(desNode2.getStart() + desNode2.getStr().split(" ").length);
                edges[entityNode1.getNodeID()][desNode2.getNodeID()].setEdgeType(Edge.TYPE_VERB);


            } else {
                // create desNode1, e.g., 'xx' in what (xx)'s
                Node desNode1 = createNode(Node.NODE_TYPE_NON_VERB);
                desNode1.setStr(matcher2.group(3).trim());
                desNode1.setStart(1);
                desNode1.setEnd(1 + desNode1.getStr().split(" ").length);
                edges[entityNode.getNodeID()][desNode1.getNodeID()].setEdgeType(Edge.TYPE_NON_VERB);

                // new desNode2, e.g. y in what (x)'s is (y)
                Node desNode2 = createNode(Node.NODE_TYPE_VERB);
                desNode2.setStr(matcher2.group(5)
                        .replace("?", "").
                                replace(".", "").
                                replace("!", "").trim());
                desNode2.setStart(desNode1.getEnd() + 1);
                desNode2.setEnd(desNode2.getStart() + desNode2.getStr().split(" ").length);
                edges[entityNode.getNodeID()][desNode2.getNodeID()].setEdgeType(Edge.TYPE_VERB);
            }

            return true;
        }

        // e.g., what is the number of.
        Matcher matcher1 = Pattern.compile("(?i)(^what is (the .*)?(number|amount) of (.*)$)").matcher(question.trim());
        if (matcher1.matches()) {
            //System.out.println(matcher.group(4));

            System.out.println("[DEBUG] SpecialSent Matcher1 Matching:" + question);
            // new root, the question type is COUNT
            Node rootNode = createNode(Node.NODE_TYPE_ROOT);
            rootNode.setTrigger("what is the number of");
            rootNode.setQuesType(QueryType.COUNT);

            // new entity, and set the edge from root to entity
            Node entityNode = createNode(Node.NODE_TYPE_ENTITY);
            edges[rootNode.getNodeID()][entityNode.getNodeID()].setEdgeType(Edge.TYPE_QUEST);

            String npString = matcher1.group(4).
                    replace("?", "").
                    replace(".", "").
                    replace("!", "");
            String tempSyntax = NLPUtil.getSyntaxTree(npString);
            //System.out.println(TransferParentheses(tempSyntax));
            TreeNode npTreeNode = createTree(tempSyntax).getFirstChild();
            if (npTreeNode.data.trim().equals("NP")) {
                NP(npTreeNode, rootNode);
            } else if (npTreeNode.data.trim().equals("S")) {
                S(npTreeNode, entityNode);
            } else {
                logger.error("SpecialSent not Cover-1:" + question);
                Node desNode = createNode(Node.NODE_TYPE_NON_VERB);
                desNode.setStr(npString);
                desNode.setStart(rootNode.getTrigger().split(" ").length);
                desNode.setEnd(taggedQuestion.split(" ").length);
                edges[entityNode.getNodeID()][desNode.getNodeID()].setEdgeType(Edge.TYPE_NON_VERB);
            }
            rootNode.setQuesType(QueryType.COUNT);
            return true;
        }

        // e.g, is it true that..
        Matcher matcher = Pattern.compile("(?i)(^((is it true (that )?)(.*$)))").matcher(question.trim());
        if (matcher.matches()) {

            System.out.println("[DEBUG] SpecialSent Matcher0 Matching:" + question);
            // a new Root, and set the question type to Judge
            Node rootNode = createNode(Node.NODE_TYPE_ROOT);
            rootNode.setTrigger("is it true");
            rootNode.setQuesType(QueryType.JUDGE);

            // new entiy node, and set the edge from root to entity
            Node entityNode = createNode(Node.NODE_TYPE_ENTITY);
            edges[rootNode.getNodeID()][entityNode.getNodeID()].setEdgeType(Edge.TYPE_QUEST);

            // S/SQ/VP
            String npString = matcher.group(5).
                    replace("?", "").
                    replace(".", "").
                    replace("!", "");
            String tempSyntax = NLPUtil.getSyntaxTree(npString);
            TreeNode STreeNode = createTree(tempSyntax).getFirstChild();
            if (STreeNode.data.trim().equals("S")) {

                if (STreeNode.getChildren().size() == 2) {
                    TreeNode npNode = STreeNode.getFirstChild();
                    TreeNode vpNode = STreeNode.getLastChild();

                    Node nvpNode = createNode(Node.NODE_TYPE_NON_VERB);
                    ComputeNodeStartEnd(nvpNode, npNode);
                    edges[entityNode.getNodeID()][nvpNode.getNodeID()].setEdgeType(Edge.TYPE_NON_VERB);

                    if (NLPUtil.judgeIfEntity(selectLeaf(npNode))) {
                        edges[entityNode.getNodeID()][nvpNode.getNodeID()].setEqual(true);
                    }

                    List<Integer> nodeIDAfterSQ = SQ(vpNode, entityNode);
                    for (Integer i : nodeIDAfterSQ) {
                        edges[entityNode.getNodeID()][i].setEdgeType(Edge.TYPE_VERB);
                    }
                }

            } else if (STreeNode.data.trim().equals("SQ") || STreeNode.data.trim().equals("VP")) {
                List<Integer> nodeIDAfterSQ = SQ(STreeNode, entityNode);
                for (Integer i : nodeIDAfterSQ) {
                    edges[entityNode.getNodeID()][i].setEdgeType(Edge.TYPE_VERB);
                }

            } else {
                logger.error("SpecialSent not Cover-2:" + question);
            }
            rootNode.setQuesType(QueryType.JUDGE);
            return true;
        }

        return false;
    }

    /**
     * the main function of parsing a question into an EDG
     *
     * @param root the root node of the syntax treeNode
     */
    private void Sent(TreeNode root) {
        // Determine if it's a special case, e.g., is it true that...
        if (!JudgeSpecialSent()) {

            // a new root
            Node node = createNode(Node.NODE_TYPE_ROOT);

            // the root of syntax tree
            if (root == null || root.children.isEmpty()) { // errors in syntax tree, EDG generation fails
                return;
            }

            // nodes under root
            TreeNode treeNode = root.children.get(0);

            boolean needRefact = false;
            boolean fragHandled = false; // if it's handled in frag module

            if (treeNode.data.trim().equals("FRAG")) { // questions generated incorrectly by CoreNLP, marked as FRAG

                // detect if it is a special question
                Matcher matcher = Pattern.compile("^(?i)((which|where|who|what|whose|when) (.*))").matcher(taggedQuestion);

                //
                if (matcher.matches()) {
                    node.setTrigger(matcher.group(2)); // trigger, e.g., which / what
                    node.setQuesType(QueryType.COMMON); // the question type is COMMON

                    // a new entity node
                    Node newNode = createNode(Node.NODE_TYPE_ENTITY);

                    // connect entity node to root
                    edges[0][newNode.getNodeID()].setEdgeType(Edge.TYPE_QUEST);

                    // CoreNLP generation has failed, generate syntax tree of the following phrase
                    String newSynTree = NLPUtil.getSyntaxTree(matcher.group(3));
                    System.out.println("partTree:" + newSynTree);  // print the syntax tree of phrase
                    TreeNode newTreeNode = createTree(newSynTree);
                    if (newTreeNode.children.size() > 0) {
                        newTreeNode = newTreeNode.children.get(0);
                    }

                    ConcateDes(newTreeNode, newNode.getNodeID(), 0);

                    // it has been handled in FRAG
                    fragHandled = true;
                }

                // detect if it is a general question
                Matcher matcher1 = Pattern.compile("^(?i)((is|was|are|were|do|did|does) (.*))").matcher(taggedQuestion);
                // it is a general question
                if (matcher1.matches()) { // general question frag
                    String trigger = matcher1.group(2);
                    node.setTrigger(trigger.trim());
                    node.setQuesType(QueryType.JUDGE);
                    // generation question, reprocessing required
                    needRefact = true;
                }

                // detect if it is a declarative sentence
                if (judgeIfImperative(taggedQuestion)) {
                    System.out.println("Imperative FRAG:" + taggedQuestion);
                    String trigger = getImperativeTrigger(taggedQuestion);
                    node.setTrigger(trigger);
                    if (judgeIfCount(question)) {
                        node.setQuesType(QueryType.COUNT);
                    } else {
                        node.setQuesType(QueryType.LIST);
                    }

                    String description = taggedQuestion.replaceAll("(?i)" + trigger, "").
                            replaceAll("\\?", "").replaceAll("\\.", "").
                            replaceAll("!", "").trim();

                    // create a new entity
                    Node newNode = createNode(Node.NODE_TYPE_ENTITY);
                    edges[0][newNode.getNodeID()].setEdgeType(Edge.TYPE_QUEST);

                    // create a description Node
                    Node desNode = createNode(Node.NODE_TYPE_NON_VERB);
                    desNode.setStr(description);
                    edges[newNode.getNodeID()][desNode.getNodeID()].setEdgeType(Edge.TYPE_NON_VERB);

                    // it has been processed in frag
                    fragHandled = true;
                }
            }

            // prefix, e.g., “Of which ... ?”
            String prefix = null;
            if (treeNode.data.equals("PP ")) { // of which...?
                prefix = selectLeaf(treeNode.children.get(0));  // get the prefix
                treeNode = treeNode.children.get(1); // the following is treat as a normal question

            }

            if (treeNode.data.trim().equals("S")) { // imperative / general question / question word in the middle

                // the string of current sub-tree
                String temp = selectLeaf(treeNode);

                // determine three situations: imperative / general question / question word in the middle
                if (judgeIfImperative(temp)) { // imperative

                    if (judgeIfCount(temp)) { // determine if it's COUNT
                        node.setQuesType(QueryType.COUNT); // the question type is COUNT
                    } else { // general question
                        node.setQuesType(QueryType.LIST); // the question type is LIST
                    }

                    String triggerWord = getImperativeTrigger(temp); // set the trigger word
                    node.setTrigger(triggerWord);

                    TreeNode treeNode1 = treeNode.getFirstChild(); // the most left child of node S
                    if (treeNode.children.size() <= 2) { // the number of nodes under S is smaller than or equal to 2
                        if (treeNode1.data.equals("VP ")) { //(VP (VB List) (NP the xx of xx..))
                            imperativeSentence(node.getNodeID(), node.getTrigger(), treeNode1);  // imperative
                        } else {
                            String restString = taggedQuestion.replaceAll("(?i)" + triggerWord, "");
                            Node entityNode = createNode(Node.NODE_TYPE_ENTITY); // new entity node
                            edges[node.getNodeID()][entityNode.getNodeID()].setEdgeType(Edge.TYPE_QUEST);

                            Node desNode = createNode(Node.NODE_TYPE_NON_VERB); // new description node
                            edges[entityNode.getNodeID()][desNode.getNodeID()].setEdgeType(Edge.TYPE_NON_VERB);
                            desNode.setStr(restString);

                        }
                    } else if (treeNode.children.size() == 3) {
                        // the number of nodes under S is 3

                        // the last one may be a period
                        TreeNode treeNode2 = treeNode.children.get(1);
                        if (treeNode1.data.equals("NP ") && treeNode2.data.equals("VP ")) {
                            String temps = selectLeaf(treeNode1).toLowerCase().replaceAll(node.getTrigger(), "");

                            // create a new entity node
                            Node node0 = createNode(Node.NODE_TYPE_ENTITY);
                            edges[node.getNodeID()][node0.getNodeID()].edgeType = Edge.TYPE_QUEST; //quest

                            // create a NPV description node
                            Node node1 = createNode(Node.NODE_TYPE_NON_VERB);

                            // tree1 as the string of NVP description node
                            ComputeNodeStartEnd(node1, treeNode1);
                            node1.setStart(node1.getEnd() - temps.split(" ").length + 2);
                            edges[node0.getNodeID()][node1.getNodeID()].edgeType = Edge.TYPE_NON_VERB; // nvp edge

                            List<Integer> list = SQ(treeNode2, node0); // nodeID list of VP description node
                            for (Integer x : list) {
                                edges[node0.getNodeID()][x].edgeType = Edge.TYPE_VERB; // vp edge
                            }
                        } else { // situations not covered, concatenate the following nodes
                            // a new Entity Node
                            System.out.println("[DEBUG] special S not covered");
                            Node entityNode = createNode(Node.NODE_TYPE_ENTITY);
                            edges[node.getNodeID()][entityNode.getNodeID()].setEdgeType(Edge.TYPE_QUEST);

                            if (taggedQuestion.matches("(?i)(^(list) (.*))")) { // list is misidentified

                                String newSyntaxTree = NLPUtil.getSyntaxTree(
                                        taggedQuestion.replaceAll("(?i)(^list)", "")
                                                .replace(".", "").trim());
                                TreeNode newTreeNode = createTree(newSyntaxTree);
                                //System.out.println(TransferParentheses(newSyntaxTree));

                                for (TreeNode tmpTreeNode : newTreeNode.getChildren()) {
                                    ConcateDes(tmpTreeNode, entityNode.getNodeID(), 0);
                                }
                            } else { // other situations
                                for (TreeNode tmpTreeNode : treeNode.children) {
                                    ConcateDes(tmpTreeNode, entityNode.getNodeID(), 0);
                                }
                            }
                        }
                    } else {
                        // the number of nodes under S >= 3
                        Node entityNode = createNode(Node.NODE_TYPE_ENTITY);
                        edges[node.getNodeID()][entityNode.getNodeID()].setEdgeType(Edge.TYPE_QUEST);

                        Node desNode = createNode(Node.NODE_TYPE_NON_VERB);
                        desNode.setStr(taggedQuestion.replaceAll("(?i)" + node.getTrigger(), "").trim());
                        edges[entityNode.getNodeID()][desNode.getNodeID()].setEdgeType(Edge.TYPE_NON_VERB);

                    }

                } else if (judgeIfGeneral(temp)) { // general question
                    // the question type is JUDGE
                    node.setQuesType(QueryType.JUDGE);

                    if (treeNode.children.size() == 2 && treeNode.children.get(0).data.equals("VP ")) {
                        // two children under S, the left one is VP, and the right one usually is '?'
                        // e.g., Does Breaking Bad have more episodes than Game of Thrones?
                        treeNode = treeNode.children.get(0); // VP node
                        // set the trigger, usually the leading be verbs / auxiliary verbs
                        node.setTrigger(selectLeaf(treeNode.children.get(0), "").toLowerCase().trim());
                        if (treeNode.children.size() < 2) {
                            // no child or only one child
                            logger.error("general question not covered:" + question);
                        } else if (treeNode.children.size() == 2) { // VP = auxiliary verb + X

                            if (treeNode.children.get(1).data.trim().equals("SBAR")) {//VBZ+SBAR
                                // Does Breaking Bad have more episodes than Game of Thrones?
                                TreeNode SBARNode = treeNode.getChildren().get(1);
                                if (!SBARNode.children.isEmpty()) {// empty list check

                                    TreeNode child = SBARNode.getChildren().get(0); // the child of SBAR, usually S
                                    List<TreeNode> toHandle = new ArrayList<>(); // node of syntax tree to be processed
                                    if (!child.getChildren().isEmpty()) {
                                        toHandle.addAll(child.getChildren());
                                    }

                                    if (!toHandle.isEmpty()) {
                                        int entityIdx = GeneralQuestion(toHandle, 0);
                                        edges[node.getNodeID()][entityIdx].edgeType = Edge.TYPE_QUEST;
                                    }
                                } else {
                                    logger.error("general question not covered: " + question);
                                }

                            } else if (treeNode.children.get(1).data.trim().equals("NP")) { // VBZ + NP
                                /*
                                  similar as SBAR
                                 */
                                TreeNode NPNode = treeNode.children.get(1);
                                List<TreeNode> toHandle = new ArrayList<>(NPNode.children); // syntax tree nodes to be processed
                                int indexofentity = GeneralQuestion(toHandle, 0);
                                edges[node.getNodeID()][indexofentity].edgeType = Edge.TYPE_QUEST;
                            }
                        } else if (treeNode.children.size() == 3) {
                            // three nodes under VP

                            TreeNode treeNode2 = treeNode.children.get(1);
                            TreeNode treeNode3 = treeNode.children.get(2);
                            List<TreeNode> listtree = new ArrayList<>();
                            listtree.add(treeNode2);
                            listtree.add(treeNode3);
                            int entityIdx = GeneralQuestion(listtree, 0);
                            edges[node.getNodeID()][entityIdx].edgeType = Edge.TYPE_QUEST;
                        }

                    }
                } else if (treeNode.children.size() == 3) {
                    // three nodes under S: SBAR + VP + XX or SBAR + SQ + XX
                    if (treeNode.children.get(0).data.trim().equals("SBAR")) {
                        if (treeNode.children.get(1).data.trim().equals("VP")
                                || treeNode.children.get(1).data.trim().equals("SQ")) {
                            // a new EntityNode
                            Node node1 = new Node();
                            node1.setNodeType(Node.NODE_TYPE_ENTITY);
                            node1.setNodeID(numNode);
                            nodes[numNode] = node1;
                            numNode++;
                            edges[0][node1.getNodeID()].edgeType = Edge.TYPE_QUEST;

                            SBAR(treeNode.children.get(0), node1, true);
                            List<Integer> nodeIdxs = SQ(treeNode.children.get(1), node1);
                            for (Integer in2 : nodeIdxs) {
                                edges[node1.getNodeID()][in2].edgeType = Edge.TYPE_VERB;
                            }
                        } else {
                            logger.error("Sent S not Cover-1:" + question);
                        }
                    } else {
                        //e.g. Greater Napanee is the home town of what people?

                        System.out.println("[DEBUG] the question word is not at the beginning");
                        List<String> posTags = NLPUtil.getPOS(taggedQuestion);
                        List<String> tokens = NLPUtil.getTokens(taggedQuestion);

                        int whIndex = -1;
                        ListIterator<String> iterator = posTags.listIterator(posTags.size());
                        while (iterator.hasPrevious()) { // find the wh-word from the back to the front
                            String previous = iterator.previous();
                            if (judgeIFWHTag(previous)) {
                                whIndex = iterator.nextIndex(); // get the last wh-word's index
                                break;
                            }
                        }
                        boolean countTag = false;
                        String whWord = tokens.get(whIndex);
                        if (whWord.toLowerCase().trim().equals("how")) { // it maybe 'how many'
                            if ((whIndex + 1) < tokens.size()) {
                                String afterWord = tokens.get(whIndex + 1).toLowerCase().trim();
                                if (afterWord.equals("many") || afterWord.equals("much")) {
                                    whWord = whWord + " " + afterWord;
                                    countTag = true;
                                }
                            }
                        }
                        node.setTrigger(whWord);
                        if (countTag) {
                            node.setQuesType(QueryType.COUNT);
                        } else {
                            node.setQuesType(QueryType.COMMON);// COMMON TYPE BY DEFAULT
                        }

                        //create a entityNode
                        Node entityNode = createNode(Node.NODE_TYPE_ENTITY);
                        edges[node.getNodeID()][entityNode.getNodeID()].setEdgeType(Edge.TYPE_QUEST);

                        //string after wh-word
                        StringBuilder afterWH = new StringBuilder();
                        int index = whIndex + 1;
                        if (countTag) {
                            index++;
                        }
                        for (; index < tokens.size(); index++) {
                            if (!tokens.get(index).trim().matches("[?!.,/]")) {
                                afterWH.append(" ").append(tokens.get(index).trim());
                            }
                        }

                        if (!afterWH.toString().trim().equals("")) {//not null
                            //create a des Node
                            Node desNode1 = createNode(Node.NODE_TYPE_NON_VERB);
                            desNode1.setStr(afterWH.toString().trim());
                            desNode1.setStart(whIndex + 1);
                            desNode1.setEnd(tokens.size());
                            edges[entityNode.getNodeID()][desNode1.getNodeID()].setEdgeType(Edge.TYPE_NON_VERB);
                        }

                        //string before wh-word
                        StringBuilder beforeWH = new StringBuilder();
                        for (int i = 0; i < whIndex; i++) {
                            beforeWH.append(" ").append(tokens.get(i).trim());
                        }
                        if (!beforeWH.toString().trim().equals("")) {//not null

                            Node desNode2 = createNode(Node.NODE_TYPE_VERB);
                            desNode2.setStr(beforeWH.toString().trim());
                            desNode2.setStart(0);
                            desNode2.setEnd(whIndex);
                            edges[entityNode.getNodeID()][desNode2.getNodeID()].setEdgeType(Edge.TYPE_VERB);

                        }

                    }
                } else if (temp.endsWith("?")) {
                    // '?' is at the end

                    // new Entity Node
                    Node node1 = createNode(Node.NODE_TYPE_ENTITY);

                    if (judgeIfSpecial(question)) { // special question
                        String trigger = getWHTrigger(question);
                        node.setTrigger(trigger);
                        if (judgeIfCount(trigger)) {
                            node.setQuesType(QueryType.COUNT);
                        } else {
                            node.setQuesType(QueryType.COMMON);
                        }
                    }

                    List<Integer> nodeIdxs = S(treeNode, node1);
                    for (Integer in1 : nodeIdxs) {
                        edges[node1.getNodeID()][in1].edgeType = Edge.TYPE_VERB; // set the edge type
                    }

                    if (edges[0][node1.getNodeID()].edgeType == Edge.TYPE_NO_EDGE) { // quest edge
                        if (!judgeIfSpecial(question)) {
                            node.setQuesType(QueryType.JUDGE); // judge
                        }
                        edges[0][node1.getNodeID()].edgeType = Edge.TYPE_QUEST;

                        if (treeNode.children.size() > 1) {
                            TreeNode treeNode1 = treeNode.children.get(0);
                            TreeNode treeNode2 = treeNode.children.get(1);
                            if (treeNode1.data.equals("NP ") && treeNode2.data.equals("VP ")) { // NP + VP
                                nodes[nodeIdxs.get(0)].setStart(getFirstLeaf(treeNode2).index);
                                nodes[nodeIdxs.get(0)].setEnd(getLastLeaf(treeNode2).index + 1);
                                List<Integer> listg2 = NP(treeNode1, node1);
                                for (Integer in2 : listg2) {
                                    edges[node1.getNodeID()][in2].edgeType = Edge.TYPE_NON_VERB;
                                }
                            }
                        }

                    }
                } else {
                    if (!SpecialDeclar(node)) {
                        logger.error("not covered S:" + temp);
                    }
                }

            } else if (treeNode.data.equals("SQ ")) { // general question / imperative sentence

                String temp = selectLeaf(treeNode, "");
                // determine if it's imperative
                if (judgeIfImperative(temp)) { // imperative sentence
                    /*
                        The same as the imperative sentence processing in S,
                        but the tag is different, one is S, and the other one is SQ
                     */
                    if (judgeIfCount(temp)) {
                        node.setQuesType(QueryType.COUNT);
                    } else {
                        node.setQuesType(QueryType.LIST);
                    }
                    node.setTrigger(getImperativeTrigger(temp));
                    if (treeNode.children.size() == 2) {
                        TreeNode treeNode1 = treeNode.children.get(0);
                        if (treeNode1.data.equals("VP ")) {
                            imperativeSentence(node.getNodeID(), node.getTrigger(), treeNode1);
                        }
                    } else {
                        logger.error("SQ not Cover:" + question);
                    }
                } else { // general question / special declarative sentence

                    if (judgeIfGeneral(taggedQuestion)) { // if it's a general question
                        int len = treeNode.children.size();
                        node.setQuesType(QueryType.JUDGE); // the question type is JUDGE
                        List<TreeNode> listtree = new ArrayList<>();
                        int lenofadvp = 0;
                        if (len > 2) { // at least 3 children
                            TreeNode treeNode1 = treeNode.children.get(0);
                            TreeNode treeNode2 = treeNode.children.get(1);
                            TreeNode treeNode3 = treeNode.children.get(2);
                            String ss = selectLeaf(treeNode1, "").toLowerCase(); // the leading question word
                            String[] t = selectLeaf(treeNode, "").toLowerCase().split(" ");
                            if (t[2].equals("there")) { // are there || is there
                                node.setTrigger(t[1] + " " + t[2]);
                            } else {
                                node.setTrigger(ss.substring(1));
                            }
                            if (len == 5) {
                                if (selectLeaf(treeNode3, "").equals("ADVP ")) {
                                    lenofadvp = treeNode3.data.split(" ").length - 1;
                                }
                            }
                            for (int i = 1; i < treeNode.children.size() - 1; i++) {
                                listtree.add(treeNode.children.get(i));
                            }
                        } else { // only one child or two children
                            node.setTrigger(selectLeaf(treeNode, "").trim().split(" ")[0].toLowerCase());// the first word is trigger
                            treeNode = treeNode.children.get(0); // the most left node under S
                            if (treeNode.data.trim().equals("VP")) { // VP + xx
                                int lent = treeNode.children.size();
                                if (lent == 2 && treeNode.children.get(1).data.trim().equals("SBAR")) {
                                    // VP has two children, and the second one is SBAR
                                    TreeNode SBARNode = treeNode.children.get(1);
                                    for (TreeNode child : SBARNode.children) {
                                        if (child.children != null && !child.children.isEmpty()) {
                                            listtree.addAll(child.children);
                                        }
                                    }
                                } else {
                                    for (int i = 1; i < treeNode.children.size(); i++) {
                                        listtree.add(treeNode.children.get(i));
                                    }
                                }
                            } else if (treeNode.data.trim().equals("SQ")) { //CoreNLP wrong identification, SQ is nested in SQ
                                //e.g. Is Ombla originiate in Croatia?
                                for (TreeNode child : treeNode.children) {
                                    if (!child.getData().startsWith("VB")) {
                                        if (child.getData().trim().equals("S")) {
                                            listtree.addAll(child.getChildren());
                                        } else {
                                            listtree.add(child);
                                        }
                                    }
                                }
                            } else {
                                for (int i = 1; i < treeNode.children.size(); i++) { // except for the most left node
                                    listtree.add(treeNode.children.get(i));
                                }
                            }
                        }
                        int indexofentity = GeneralQuestion(listtree, lenofadvp);
                        edges[node.getNodeID()][indexofentity].edgeType = Edge.TYPE_QUEST;
                    } else { // special declarative sentence
                        SpecialDeclar(node);
                    }
                }
            } else if (treeNode.data.trim().equals("SBARQ") || treeNode.data.trim().equals("SBAR")) { // SBARQ or SBAR
                if (treeNode.children.size() < 2) { // the number of children of SBARQ < 2
                    logger.error("Sent.SBARQ not Cover-1:" + question);
                } else if (treeNode.children.size() >= 4) {// the children of SBARQ >=4
                    // new entity node
                    Node entityNode = createNode(Node.NODE_TYPE_ENTITY);
                    edges[node.getNodeID()][entityNode.getNodeID()].setEdgeType(Edge.TYPE_QUEST);

                    if (treeNode.getFirstChild().data.trim().startsWith("WH")) {
                        // set the trigger and the question type
                        String triggerString = selectLeaf(treeNode.getFirstChild());
                        String whWord = getWHWord(triggerString);
                        node.setQuesType(QueryType.COMMON);
                        node.setTrigger(whWord);

                        // what film did...
                        String targetType = triggerString.replaceAll("(?i)" + whWord, "").trim();
                        if (!targetType.equals("")) {
                            Node desNode1 = createNode(Node.NODE_TYPE_NON_VERB);
                            desNode1.setStr(targetType);
                            edges[entityNode.getNodeID()][desNode1.getNodeID()].setEdgeType(Edge.TYPE_NON_VERB);
                        }
                        StringBuilder vpdes = new StringBuilder();
                        for (int i = 1; i < treeNode.children.size(); i++) {
                            vpdes.append(" ").append(selectLeaf(treeNode.children.get(i)).trim());
                        }
                        Node desNode2 = createNode(Node.NODE_TYPE_VERB);
                        desNode2.setStr(vpdes.toString().trim());
                        edges[entityNode.getNodeID()][desNode2.getNodeID()].setEdgeType(Edge.TYPE_VERB);

                    } else {
                        String firstTreeString = selectLeaf(treeNode.getFirstChild());
                        String whWord = getWHWord(firstTreeString);

                        node.setQuesType(QueryType.COMMON);
                        node.setTrigger(whWord);

                        String remainPart = firstTreeString.replaceAll("(?i)" + whWord, "").trim();

                        StringBuilder vpdes = new StringBuilder(remainPart);
                        for (int i = 1; i < treeNode.children.size(); i++) {
                            vpdes.append(" ").append(selectLeaf(treeNode.children.get(i)).trim());
                        }
                        Node desNode2 = createNode(Node.NODE_TYPE_VERB);
                        desNode2.setStr(vpdes.toString().trim());
                        edges[entityNode.getNodeID()][desNode2.getNodeID()].setEdgeType(Edge.TYPE_VERB);
                    }

                } else {
                    // the number of children of SBARQ / SBAR is 2-3

                    // new Entity Node
                    Node node0 = createNode(Node.NODE_TYPE_ENTITY);

                    TreeNode treeNode1 = treeNode.children.get(0); // first Node
                    TreeNode treeNode2 = treeNode.children.get(1); // Second Node
                    if (treeNode1.data.equals("WHPP ")) { // preposition + which / how / ...
                        String tempString = selectLeaf(treeNode1.children.get(1), "");
                        if (tempString.trim().startsWith("how")) { // of how ...
                            node.setQuesType(QueryType.COUNT); // count
                            String[] tempString1 = selectLeaf(treeNode1.children.get(1), "").trim().toLowerCase().split(" ");
                            node.setTrigger(tempString1[1] + " " + tempString1[2]); // how many / how much
                        } else {
                            node.setQuesType(QueryType.COMMON); // common
                            try {
                                node.setTrigger(selectLeaf(treeNode1, "").trim().toLowerCase().split(" ")[2]);
                            } catch (ArrayIndexOutOfBoundsException e) {
                                e.printStackTrace();
                            }
                        }
                        if (!treeNode1.children.get(1).data.equals("WHNP ")) { // tree1 under WHPP is not WHNP
                            logger.error("Sent SBAR not Cover-3:" + question);

                        }

                        int in1 = WH(treeNode1.children.get(1), node0);
                        edges[node.getNodeID()][in1].edgeType = 1;
                        List<Integer> listg2 = SQ(treeNode2, nodes[in1]);
                        for (Integer in2 : listg2) {
                            edges[in1][in2].edgeType = Edge.TYPE_VERB;
                            edges[in1][in2].info = selectLeaf(treeNode1.children.get(0), "").substring(1).toLowerCase();
                        }
                    } else if (selectLeaf(treeNode1, "").trim().toLowerCase().startsWith("when") || selectLeaf(treeNode1, "").trim().toLowerCase().startsWith("where")) {
                        node.setQuesType(QueryType.COMMON); // question type
                        // set trigger
                        if (treeNode1.children.get(0).str != null) {
                            node.setTrigger(treeNode1.children.get(0).str.substring(1));
                        } else {
                            node.setTrigger(selectLeaf(treeNode1.children.get(0), "").split(" ")[1]);
                        }

                        int in1 = WH(treeNode1, node0);
                        edges[node.getNodeID()][in1].edgeType = Edge.TYPE_QUEST;// set edge type

                        List<Integer> listg2 = SQ(treeNode2, nodes[in1]);
                        for (Integer in2 : listg2) {
                            edges[in1][in2].edgeType = Edge.TYPE_VERB;
                        }
                    } else if (treeNode1.data.startsWith("WHNP")) {
                        if (selectLeaf(treeNode1).trim().toLowerCase().startsWith("how many")) {
                            node.setQuesType(QueryType.COUNT);
                            node.setTrigger("how many");
                            int in1 = WH(treeNode1, node0);
                            edges[node.getNodeID()][in1].edgeType = Edge.TYPE_QUEST;
                            List<Integer> listg2 = SQ(treeNode2, nodes[in1]);
                            for (Integer in2 : listg2) {
                                edges[in1][in2].edgeType = Edge.TYPE_VERB;
                            }
                        } else {
                            node.setQuesType(QueryType.COMMON); // common
                            node.setTrigger(selectLeaf(treeNode1, "").split(" ")[1].toLowerCase()); // set trigger
                            int in1 = WH(treeNode1, node0);//index of entity
                            edges[node.getNodeID()][in1].edgeType = Edge.TYPE_QUEST;

                            if (treeNode2.data.equals("S ") && treeNode2.children.size() == 1) { // sometimes SQ maybe under S
                                treeNode2 = treeNode2.children.get(0);
                            }
                            List<Integer> listg2 = SQ(treeNode2, nodes[in1]);
                            for (Integer in2 : listg2) {
                                edges[in1][in2].edgeType = Edge.TYPE_VERB; // verb
                            }
                        }
                    } else if (selectLeaf(treeNode1, "").toLowerCase().trim().startsWith("how")) { // How + adj./adv.(sometimes not directly followed by a noun)
                        edges[node.getNodeID()][node0.getNodeID()].edgeType = Edge.TYPE_QUEST;
                        node.setTrigger(selectLeaf(treeNode1, "").substring(1));

                        if (treeNode1.data.trim().equals("SBAR") && ((node.getTrigger().toLowerCase().startsWith("how many") || node.getTrigger().toLowerCase().startsWith("how much")))) { // SBAR = WHADJP + S
                            node.setTrigger(selectLeaf(treeNode1.getFirstChild()).trim());
                            node.setQuesType(QueryType.COUNT);

                            Node node2 = createNode(Node.NODE_TYPE_VERB);
                            node2.setStr(taggedQuestion.replaceAll("(?i)" + node.getTrigger(), "").trim());
                            node2.setStart(2);
                            node2.setEnd(taggedQuestion.split(" ").length);
                            edges[node0.getNodeID()][node2.getNodeID()].edgeType = Edge.TYPE_VERB;
                        } else {
                            if ((treeNode1.data.trim().equals("WHNP")) && (node.getTrigger().toLowerCase().startsWith("how many") || node.getTrigger().toLowerCase().startsWith("how much"))) {
                                node.setTrigger(selectLeaf(treeNode1.getFirstChild()).trim());
                                node.setQuesType(QueryType.COUNT);
                                Node node2 = createNode(Node.NODE_TYPE_NON_VERB);
                                ComputeNodeStartEnd(node2, treeNode1);
                                node2.setStart(node2.getStart() + 2);
                                edges[node0.getNodeID()][node2.getNodeID()].edgeType = Edge.TYPE_NON_VERB;

                            } else {
                                if (node.getTrigger().toLowerCase().contains("how many") || node.getTrigger().toLowerCase().contains("how much")) {
                                    node.setQuesType(QueryType.COUNT);
                                } else {
                                    node.setQuesType(QueryType.EXTENT);
                                }
                            }
                            List<Integer> listg2 = SQ(treeNode2, node0);
                            for (Integer in2 : listg2) {
                                edges[node0.getNodeID()][in2].edgeType = Edge.TYPE_VERB;
                            }
                        }
                    } else if (treeNode1.data.startsWith("WHNP")) {
                        node.setQuesType(QueryType.COMMON); // common
                        node.setTrigger(selectLeaf(treeNode1, "").split(" ")[1].toLowerCase());
                        int in1 = WH(treeNode1, node0); // index of entity
                        edges[node.getNodeID()][in1].edgeType = 1;
                        if (treeNode2.data.equals("S ") && treeNode2.children.size() == 1) {
                            treeNode2 = treeNode2.children.get(0);
                        }
                        List<Integer> listg2 = SQ(treeNode2, nodes[in1]);
                        for (Integer in2 : listg2) {
                            edges[in1][in2].edgeType = 3; // verb
                        }
                    } else if (treeNode1.data.equals("SBARQ ")) {
                        Sent(treeNode1);
                    } else if (treeNode1.data.equals("SBAR ")) {
                        edges[node.getNodeID()][node0.getNodeID()].edgeType = 1;
                        List<Integer> listg1 = SBAR(treeNode1, node0, true);
                        if (treeNode2.data.equals("SQ ")) {
                            List<Integer> listg2 = SQ(treeNode2, node0);
                            for (int i = 0; i < listg2.size(); i++) {
                                edges[node0.getNodeID()][listg2.get(i)].edgeType = 3;
                            }
                        }
                    } else {
                        //  System.out.println("error in Sent");
                        int in1 = WH(treeNode1, node0);
                        List<Integer> listg2 = SQ(treeNode2, nodes[in1]);
                        for (Integer in2 : listg2) {
                            edges[in1][in2].edgeType = 3;
                        }
                    }
                }
            } else if (treeNode.data.trim().equals("SINV") || needRefact) {
                // inverted declarative sentence / general question, e.g., Was winston churchill the prime minister of Selwyn Lloyd
                if (needRefact) {
                    System.out.println("FRAG to SINV question:" + question);
                } else {
                    System.out.println("SINV question:" + question);
                }

                if (judgeIfGeneral(question)) {
                    String tig = getGeneralTrigger(question);
                    if (question.toLowerCase().trim().startsWith(Objects.requireNonNull(tig))) {
                        node.setQuesType(QueryType.JUDGE); //judge
                        node.setTrigger(tig);

                        //create entity node
                        Node node0 = createNode(Node.NODE_TYPE_ENTITY);
                        edges[node.getNodeID()][node0.getNodeID()].edgeType = 1;
                        int entityNodeID = node0.getNodeID();

                        String tempQuestion = selectLeaf(treeNode);
                        tempQuestion = tempQuestion.trim().replace("-LRB-", "(").replace("-RRB-", ")");

                        System.out.println("tempQuestion:" + tempQuestion);
                        Pattern pattern = Pattern.compile("(?i)(" + tig + " (.*))");
                        //System.out.println("pattern:"+pattern.toString());
                        Matcher matcher = pattern.matcher(tempQuestion);
                        if (matcher.matches()) {
                            //System.out.println("matcher:"+matcher.group(2));
                            tempQuestion = matcher.group(2);
                        }
                        System.out.println("tempQuestion:" + tempQuestion);

                        Pattern pattern1 = Pattern.compile("^(<e\\d>|the <e\\d>) (.*)");
                        //System.out.println("pattern1:" + pattern1);
                        Matcher matcher1 = pattern1.matcher(tempQuestion.trim());

                        String firstNode = null;
                        String secondNode = null;
                        int firstStart = 0;
                        int firstEnd = 0;
                        int secondStart = 0;
                        int secondEnd = 0;

                        if (matcher1.matches()) {
                            //System.out.println("group1:" + matcher1.group(1));
                            //System.out.println("group2:" + matcher1.group(2));
                            firstNode = matcher1.group(1);
                            secondNode = matcher1.group(2);
                        } else {
                            firstNode = tempQuestion;
                            String syntaxTree = NLPUtil.transferParentheses(NLPUtil.getSyntaxTree(tempQuestion));
                            System.out.println(syntaxTree);
                            TreeNode rootNode = createTree(syntaxTree);
                            LinkedList<TreeNode> toSearch = new LinkedList<>();
                            toSearch.add(rootNode);

                            while (!toSearch.isEmpty()) {
                                TreeNode pop = toSearch.pop();
                                if (pop.children.size() > 1) { // find the first forked node
                                    firstNode = selectLeaf(pop.children.get(0));
                                    firstStart = getFirstLeaf(pop.children.get(0)).index;
                                    firstEnd = getLastLeaf(pop.children.get(0)).index;

                                    secondStart = getFirstLeaf(pop).index;
                                    secondEnd = getLastLeaf(pop).index;

                                    secondNode = "";
                                    for (int i = 1; i < pop.children.size(); i++) {
                                        secondNode += selectLeaf(pop.children.get(i)).trim() + " ";
                                    }
                                    secondNode = secondNode.trim();
                                    break;
                                } else if (pop.children.size() == 1) {
                                    toSearch.addAll(pop.children);
                                }
                            }
                            System.out.println("FirstNode:" + firstNode);
                            System.out.println("SecondNode:" + secondNode);
                        }


                        // the first entity
                        Node newNode = createNode(Node.NODE_TYPE_NON_VERB);
                        newNode.setStart(firstStart);
                        newNode.setEnd(firstEnd);
                        // set the equal edge
                        edges[entityNodeID][newNode.getNodeID()].setEdgeType(Edge.TYPE_NON_VERB);
                        edges[entityNodeID][newNode.getNodeID()].setEqual(true);

                        if (secondNode != null) {
                            // the second entity
                            Node newNode1 = new Node();
                            newNode1.setStr(secondNode.trim());
                            if (NLPUtil.judgeIfVP(newNode1.getStr())) {
                                newNode1.setNodeType(3);
                            } else {
                                newNode1.setNodeType(4);
                            }
                            newNode1.setNodeID(numNode);
                            newNode1.setStart(secondStart);
                            newNode1.setEnd(secondEnd);
                            nodes[numNode] = newNode1;
                            numNode++;

                            Edge newEdge1 = new Edge();
                            newEdge1.edgeType = newNode1.getNodeType() == 3 ? 3 : 4;
                            edges[entityNodeID][newNode1.getNodeID()] = newEdge1;

                        }
                    }
                } else if (judgeIfImperative(question)) {
                    if (judgeIfCount(question)) {
                        node.setQuesType(QueryType.COUNT);
                    } else { // general imperative sentence
                        node.setQuesType(QueryType.LIST);
                    }
                    // set triggerWord
                    String triggerWord = getImperativeTrigger(question);
                    node.setTrigger(triggerWord);
                    // new entityNode
                    Node entityNode = createNode(Node.NODE_TYPE_ENTITY);
                    edges[node.getNodeID()][entityNode.getNodeID()].setEdgeType(Edge.TYPE_QUEST);
                    // new desNode
                    Node desNode = createNode(Node.NODE_TYPE_NON_VERB);
                    desNode.setStr(question.replaceAll("(?i)" + triggerWord, ""));
                    edges[entityNode.getNodeID()][desNode.getNodeID()].setEdgeType(Edge.TYPE_NON_VERB);

                }


            } else if (question.toLowerCase().startsWith("show me ")) {
                node.setQuesType(QueryType.LIST);
                Node node0 = new Node();
                node0.setNodeID(numNode);
                nodes[numNode] = node0;
                numNode++;
                node0.setNodeType(2);
                edges[node.getNodeID()][node0.getNodeID()].edgeType = 1;
                question = question.substring(8);
                syntaxTreeText = NLPUtil.getSyntaxTree(question);
                syntaxTreeNode = createTree(syntaxTreeText);
                TreeNode treeNode0 = syntaxTreeNode.children.get(0);
                if (treeNode0.data.equals("NP ") && treeNode0.children.size() == 2) {
                    if (treeNode0.children.get(0).data.equals("NP ")) {
                        NP(treeNode0.children.get(0), node0);
                    }
                }
            } else { // default

                if (!fragHandled) { //frag not handled successfully, neither do SINV
                    SpecialDeclar(node); // special declarative
                    node.setQuesType(QueryType.COMMON);//default COMMON
                }
            }
            if (prefix != null) {
                nodes[0].setTrigger(prefix + " " + nodes[0].getTrigger());
            }
            if (question.toLowerCase().contains("how many ")) {
                nodes[0].setQuesType(QueryType.COUNT);
            }
        }
    }

    /**
     * Process special declarative sentences
     *
     * @param node
     * @return
     */
    private boolean SpecialDeclar(Node node) {

        System.out.println("SpecialDeclar:" + question);

        boolean handled = false;
        // question word is at the middle, e.g., xxx is the wife of what ?
        System.out.println("Special Declar :" + taggedQuestion);

        Pattern pattern2 = Pattern.compile("(?i)((.*) (what|which|when|where|whose|whom|how many|how much)(.*))");
        Matcher matcher2 = pattern2.matcher(taggedQuestion);
        //System.out.println("questionToMatch:" + taggedQuestion);
        if (matcher2.matches()) { // the question word is after the declarative
            handled = true;
            System.out.println("Special Declar Matches matcher2");
            node.setTrigger(matcher2.group(3).trim());
            if (node.getTrigger().startsWith("how")) {
                node.setQuesType(QueryType.COUNT);
            } else {
                node.setQuesType(QueryType.COMMON); // common
            }

            // new entityNode and set quest edge
            Node entityNode = createNode(Node.NODE_TYPE_ENTITY);
            edges[node.getNodeID()][entityNode.getNodeID()].setEdgeType(Edge.TYPE_QUEST);

            String preStr = matcher2.group(2).trim();
            String postStr = matcher2.group(4).trim();

            // the part after question word
            if (!postStr.equals("")) {
                TreeNode postTreeNode = createTree(NLPUtil.getSyntaxTree(postStr)).getFirstChild();
                ConcateDes(postTreeNode, entityNode.getNodeID(), 0);
            }
            // the part before question word
            if (!preStr.equals("")) {
                TreeNode preTreeNode = createTree(NLPUtil.getSyntaxTree(preStr)).getFirstChild();
                ConcateDes(preTreeNode, entityNode.getNodeID(), 0);
            }
            return handled;
        }

        // List may be treat as NP incorrectly
        Matcher matcher = Pattern.compile("(?i)(^(list all|give me|show me|tell me|list|give|show|tell) (.*))").matcher(taggedQuestion);
        if (matcher.matches()) {
            System.out.println("Special Declar Matches matcher1");
            handled = true;
            node.setQuesType(QueryType.LIST);
            node.setTrigger(matcher.group(2));

            // new entityNode
            Node entityNode = createNode(Node.NODE_TYPE_ENTITY);
            edges[node.getNodeID()][entityNode.getNodeID()].setEdgeType(Edge.TYPE_QUEST);

            // new desNode
            Node desNode = createNode(Node.NODE_TYPE_NON_VERB);
            desNode.setStr(matcher.group(3));
            edges[entityNode.getNodeID()][desNode.getNodeID()].setEdgeType(Edge.TYPE_NON_VERB);
            return handled;
        }
        return handled;

    }

    /**
     * create a node and add it to the array
     *
     * @param nodeType the type of the new node
     * @return the new node
     */
    private Node createNode(int nodeType) {
        Node node = new Node();
        node.setNodeType(nodeType);
        node.setNodeID(numNode);
        nodes[numNode] = node;
        numNode++;

        if (nodeType == Node.NODE_TYPE_ENTITY) {
            node.setEntityID(findEntityBlockID(node));
        }

        return node;
    }

    /**
     * delete the specified node, and move forward the following nodes
     *
     * @param nodeID the nodeId of the node to be removed
     * @return the deleted index
     */
    private int deleteNode(int nodeID) {

        for (int i = nodeID; i < numNode - 1; i++) {
            nodes[i] = nodes[i + 1];
            nodes[i].setNodeID(i);
            nodes[i + 1] = null;
        }

        // remove all the edges connected to nodeID
        for (int i = 0; i < numNode; i++) {
            for (int j = 0; j < numNode; j++) {
                if ((i == nodeID || j == nodeID) && edges[i][j] != null && edges[i][j].getEdgeType() > Edge.TYPE_NO_EDGE) {
                    edges[i][j].setEdgeType(Edge.TYPE_NO_EDGE);
                }
            }
        }

        // modify all edges after the node
        for (int i = nodeID + 1; i < numNode; i++) {
            for (int j = 0; j < numNode; j++) {
                if (edges[i][j] != null && edges[i][j].getEdgeType() > Edge.TYPE_NO_EDGE) {
                    edges[i - 1][j] = edges[i][j]; // move forward
                    edges[i][j] = new Edge(); // clear current position
                }
            }

            for (int j = 0; j < numNode; j++) {
                if (edges[j][i] != null && edges[j][i].getEdgeType() > Edge.TYPE_NO_EDGE) {
                    edges[j][i - 1] = edges[j][i]; // move forward
                    edges[j][i] = new Edge(); // clear current position
                }
            }
        }
        numNode--;

        return nodeID;
    }

    /**
     * Get description nodes (verb or non-verb) of an entity
     *
     * @param entityID entityID
     * @return associated descriptive node list
     */
    public List<Node> getRelatedDescription(int entityID) {
        List<Node> result = new ArrayList<>();

        int NodeIdOfEntity = getNodeIdByEntityId(entityID);
        if (NodeIdOfEntity < 0) // index out of bound
            return result;
        for (int i = 0; i < numNode; i++) {
            if (edges[NodeIdOfEntity][i] == null) continue;
            if (edges[NodeIdOfEntity][i].edgeType == Edge.TYPE_VERB || edges[NodeIdOfEntity][i].edgeType == Edge.TYPE_NON_VERB) {
                result.add(nodes[i]);
            }
        }
        return result;
    }

    /**
     * Find the specified nodeID of a given entityID
     *
     * @param entityID entityID
     * @return the nodeID of this entity node, -1 if not found
     */
    public int getNodeIdByEntityId(int entityID) {
        for (int i = 0; i < numNode; i++) {
            if (nodes[i].getNodeType() == Node.NODE_TYPE_ENTITY && nodes[i].getEntityID() == entityID) {
                return i;
            }
        }
        return -1;
    }

    /**
     * Determine the existence of the reference between an entity and the description,
     * i.e., the determine if the entity is at the bottom of the tree
     *
     * @param entityID entityID
     * @return boolean value
     */
    public boolean entityNodeHasRefer(int entityID) {

        List<Node> relatedDes = getRelatedDescription(entityID);
        for (Node desNode : relatedDes) {
            if (desNode.isContainsRefer()) {
                return true;
            }
        }
        return false;
    }

    /**
     * Get the collection of the referred entities given an entityID (one-hop)
     *
     * @param entityID entityID
     * @return a collection of entityIDs referred by the given entity
     */
    public List<Integer> getReferredEntity(int entityID) {

        List<Integer> result = new ArrayList<>();
        if (!entityNodeHasRefer(entityID)) {  // no reference, return an empty list
            return result;
        } else {
            List<Node> relatedDes = getRelatedDescription(entityID);
            for (Node desNode : relatedDes) {
                if (desNode.isContainsRefer()) { // this description refers to some entity
                    for (int i = 0; i < numNode; i++) {// find the referring entityID
                        System.out.print(edges[desNode.getNodeID()][i].edgeType);
                        if (edges[desNode.getNodeID()][i].edgeType == Edge.TYPE_REFER) {
                            result.add(nodes[i].getEntityID());
                        }
                    }
                }
            }
        }
        return result;
    }

    /**
     * return the entityID of the entity referred from this desNode
     *
     * @param nodeID nodeID of the desNode
     * @return entityID referred from this desNode
     */
    public int getDesReferredEntityID(int nodeID) {
        for (int i = 0; i < numNode; i++) {
            if (edges[nodeID][i] != null && edges[nodeID][i].getEdgeType() == Edge.TYPE_REFER) {
                return findEntityBlockID(nodes[i]);
            }
        }
        return 0;   // return 0 by default
    }

    /**
     * judge whether this desNode is an equal Node
     *
     * @param nodeID des NodeID
     * @return whether this desNode is an equal node
     */
    public boolean isEqualNode(int nodeID) {
        return edges[findEntityNodeID(nodes[nodeID])][nodeID].isEqual();
    }

    /**
     * Get the number of descriptive nodes
     *
     * @return the number of descriptive nodes
     */
    public int getNumDescriptiveNode() {
        return (int) IntStream.range(0, numNode).filter(i -> nodes[i].getNodeType() == Node.NODE_TYPE_NON_VERB
                || nodes[i].getNodeType() == Node.NODE_TYPE_VERB).count();
    }

    /**
     * flatten an EDG to one description, which means donot decompose the question
     *
     * @return flatten an EDG to one description
     */
    public EDG flattenEDG() {
        EDG newEDG = new EDG();

        // copy the question type

        Node rootNode = newEDG.createNode(Node.NODE_TYPE_ROOT);
        Node entityNode = newEDG.createNode(Node.NODE_TYPE_ENTITY);
        Node desNode = newEDG.createNode(Node.NODE_TYPE_VERB);


        newEDG.edges[0][1].setEdgeType(Edge.TYPE_QUEST);
        newEDG.edges[1][2].setEdgeType(Edge.TYPE_VERB);

        rootNode.setQuesType(nodes[0].getQueryType());
        entityNode.setEntityID(0);
        desNode.setStr(question);
        desNode.setEntityID(0);

        newEDG.numNode = 3;
        newEDG.numEntity = 1;

        return newEDG;
    }

    /**
     * convert a block to a string
     *
     * @param entityID blockID
     * @return a sequence like [BLK][DES]...[DES]...[BLK][DES]...[DES]...
     */
    public String blockToString(int entityID) {
        StringBuilder result = new StringBuilder(" [BLK] ");
        List<Node> relatedDescription = getRelatedDescription(entityID);
        relatedDescription.forEach(node -> result.append(" [DES] ").append(node.getOriginStr()));
        for (Node desNode : relatedDescription) {
            if (desNode.isContainsRefer()) {
                int desReferredEntityID = getDesReferredEntityID(desNode.getNodeID());
                result.append(blockToString(desReferredEntityID));
            }
        }


        if (QAArgs.getDataset() == DatasetEnum.QALD_9) {
            if (entityID == 0) {
                Trigger trigger = getTrigger();
                if (trigger == Trigger.HOW) {
                    result.insert(0, "How");
                } else if (trigger == Trigger.WHEN) {
                    result.insert(0, "When");
                } else if (trigger == Trigger.WHERE) {
                    result.insert(0, "Where");
                } else if (trigger == Trigger.WHO) {
                    result.insert(0, "Who");
                } else if (trigger == Trigger.IS) {
                    result.insert(0, "Is");
                } else if (trigger == Trigger.UNKNOWN) {
                    result.insert(0, "Which");
                }
            } else {
                result.insert(0, "Which");
            }
        }

        return result.toString();

    }

    public static void main(String[] args) throws IOException {
        EDG.init(DatasetEnum.LC_QUAD);
        String question = "Which city's foundeer is John Forbes?";
        EDG e = new EDG(question);
        System.out.println(e);
    }

}