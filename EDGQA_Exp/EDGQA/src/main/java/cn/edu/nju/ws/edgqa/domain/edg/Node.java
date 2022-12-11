package cn.edu.nju.ws.edgqa.domain.edg;

import cn.edu.nju.ws.edgqa.domain.beans.TreeNode;
import cn.edu.nju.ws.edgqa.utils.enumerates.QueryType;
import org.jetbrains.annotations.NotNull;
import org.json.JSONObject;

import java.util.HashMap;
import java.util.Map;

public class Node {
    public static final int NODE_TYPE_ROOT = 1;
    public static final int NODE_TYPE_ENTITY = 2;
    public static final int NODE_TYPE_VERB = 3;
    public static final int NODE_TYPE_NON_VERB = 4;
    public static final Map<Integer, String> nodeTypeMap = new HashMap<>();
    public static final Map<Integer, String> questionTypeMap = new HashMap<>();

    static {
        nodeTypeMap.put(1, "root");
        nodeTypeMap.put(2, "entity");
        nodeTypeMap.put(3, "VP-des");
        nodeTypeMap.put(4, "nonVP-des");
        nodeTypeMap.put(0, "null");
    }

    static {
        questionTypeMap.put(1, "count");
        questionTypeMap.put(2, "judge");
        questionTypeMap.put(3, "list");
        questionTypeMap.put(4, "common");
        questionTypeMap.put(5, "extent");
    }

    private int nodeID;
    private int nodeType; // 1-root;2-entity;3-verb;4-non-verb
    //only for NodeType=ROOT
    private QueryType quesType;
    private String trigger; // trigger that indicates the question type
    //only for NodeType=VERB|NON_VERB
    private int start;
    private int end;
    private TreeNode treeNode;
    private String str;
    private String originStr;
    private boolean containsRefer;
    //only for NodeType=ENTITY|VERB|NON_VERB
    private int entityID;//i.e. entityBlockID

    /**
     * copy constructor
     *
     * @param node an EDG node
     */
    public Node(Node node) {
        this.nodeID = node.nodeID;
        this.nodeType = node.nodeType;
        this.quesType = node.quesType;
        this.trigger = node.trigger;
        this.start = node.start;
        this.end = node.end;
        this.treeNode = node.treeNode;
        this.str = node.str;
        this.containsRefer = node.containsRefer;
        this.entityID = node.entityID;

    }

    public Node() {
        nodeID = -1;
        start = -1;
        end = -1;
        nodeType = 0;
        quesType = QueryType.UNKNOWN;
        trigger = null;
        entityID = -1;
        treeNode = null;
        str = null;
        containsRefer = false;

    }

    public static Map<Integer, String> getNodeTypeMap() {
        return nodeTypeMap;
    }

    public static Map<Integer, String> getQuestionTypeMap() {
        return questionTypeMap;
    }

    public static Node fromJSON(@NotNull JSONObject o) {
        Node edg_node = new Node();
        edg_node.nodeID = o.getInt("nodeID");
        edg_node.nodeType = o.getInt("nodeType");
        edg_node.entityID = o.getInt("entityID");
        edg_node.quesType = o.getEnum(QueryType.class, "questionType");
        if (o.keySet().contains("trigger")) {
            edg_node.trigger = o.getString("trigger");
        }
        edg_node.start = o.getInt("start");
        edg_node.end = o.getInt("end");
        edg_node.containsRefer = o.getBoolean("containsRefer");
        if (o.keySet().contains("str")) {
            edg_node.str = o.getString("str");
        }

        if (o.keySet().contains("originStr")) {
            edg_node.originStr = o.getString("originStr");
        } else {
            edg_node.originStr = edg_node.str;
        }

        return edg_node;
    }

    public static String nodeType2String(int nType) {
        switch (nType) {
            case NODE_TYPE_ROOT:
                return "ROOT";
            case NODE_TYPE_ENTITY:
                return "Entity";
            case NODE_TYPE_VERB:
                return "VerbDescription";
            case NODE_TYPE_NON_VERB:
                return "NonVerbDescription";
            default:
                return null;
        }
    }

    public String getStr() {
        return str;
    }

    public void setStr(String str) {
        this.str = str;
    }

    public TreeNode getTree() {
        return treeNode;
    }

    public void setTree(TreeNode treeNode) {
        this.treeNode = treeNode;
    }

    public boolean isContainsRefer() {
        return containsRefer;
    }

    public void setContainsRefer(boolean containsRefer) {
        this.containsRefer = containsRefer;
    }

    public int getEntityID() {
        return entityID;
    }

    public void setEntityID(int entityID) {
        this.entityID = entityID;
    }

    public int getNodeID() {
        return nodeID;
    }

    public void setNodeID(int nodeID) {
        this.nodeID = nodeID;
    }

    public int getNodeType() {
        return nodeType;
    }

    public void setNodeType(int nodeType) {
        this.nodeType = nodeType;
    }

    public QueryType getQueryType() {
        return quesType;
    }

    public void setQuesType(QueryType quesType) {
        this.quesType = quesType;
    }

    public String getTrigger() {
        return trigger;
    }

    public void setTrigger(String trigger) {
        this.trigger = trigger;
    }

    public int getStart() {
        return start;
    }

    public void setStart(int start) {
        this.start = start;
    }

    public int getEnd() {
        return end;
    }

    public void setEnd(int end) {
        this.end = end;
    }

    @Override
    public String toString() {
        String nodeString = "Node{" +
                "nodeID=" + nodeID +
                ", nodeType=" + nodeTypeMap.get(nodeType);

        if (nodeType == NODE_TYPE_ROOT) { // root
            nodeString += ", questionType=" + quesType;
            nodeString += ", trigger='" + trigger + '\'';
        }

        if (nodeType == NODE_TYPE_VERB || nodeType == NODE_TYPE_NON_VERB) { // description
            nodeString += ", start=" + start +
                    ", end=" + end +
                    ", str='" + str + '\'';
            if (containsRefer) {
                nodeString += ", containRefer=" + containsRefer;
            }
        }
        if (nodeType >= NODE_TYPE_ENTITY) {
            nodeString += ", entityID=" + entityID;
        }
        nodeString += '}';
        return nodeString;
    }

    public JSONObject toJSON() {
        JSONObject edg_node = new JSONObject();
        edg_node.put("nodeID", nodeID);
        edg_node.put("nodeType", nodeType);
        edg_node.put("entityID", entityID);
        edg_node.put("questionType", quesType);
        edg_node.put("trigger", trigger);
        edg_node.put("start", start);
        edg_node.put("end", end);
        edg_node.put("containsRefer", containsRefer);
        edg_node.put("originStr", originStr);
        if (str != null) {
            edg_node.put("str", str);
        }
        return edg_node;
    }

    public String getOriginStr() {
        return originStr;
    }

    public void setOriginStr(String originStr) {
        this.originStr = originStr;
    }
}
