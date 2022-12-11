package cn.edu.nju.ws.edgqa.domain.edg;

import org.json.JSONObject;

public class Edge {
    public static final int TYPE_QUEST = 1;
    public static final int TYPE_REFER = 2;
    public static final int TYPE_VERB = 3;
    public static final int TYPE_NON_VERB = 4;
    public static final int TYPE_NO_EDGE = 0;

    public int edgeType; // 1-quest; 2-refer; 3-verb; 4-non-verb; (5-entity_to_entity;) 0-no-edge
    public int from; // from node index
    public int to; // to node index

    /**
     * Only if this edge is TYPE_REFER
     */
    public int start;
    /**
     * Only if this edge is TYPE_REFER
     */
    public int end;

    /**
     * Only if this edge is TYPE_VERB
     */
    public String info;

    /**
     * Only if this edge is TYPE_NON_VERB.
     */
    public boolean isEqual; // if it is an equal edge, for JUDGE questions


    public Edge() {
        from = -1;
        to = -1;
        start = -1;
        end = -1;
        edgeType = 0;
        info = null;
        isEqual = false;
        // relation = null;
    }

    public static String edgeType2String(int etype) {

        switch (etype) {
            case 1:
                return "Quest";
            case 2: {
                return "Refer";
            }
            case 3: {
                return "VP Const.";
            }
            case 4: {
                return "Non-VP Const.";
            }
            default:
                return null;
        }

    }

    public static Edge fromJSON(JSONObject edgeJSON) {
        Edge edge = new Edge();
        edge.from = edgeJSON.getInt("from");
        edge.to = edgeJSON.getInt("to");
        edge.start = edgeJSON.getInt("start");
        edge.end = edgeJSON.getInt("end");
        edge.edgeType = edgeJSON.getInt("edgeType");
        edge.isEqual = edgeJSON.getBoolean("isEqual");
        if (edgeJSON.keySet().contains("info")) {
            edge.info = edgeJSON.getString("info");
        }
        return edge;

    }

    public int getEdgeType() {
        return edgeType;
    }

    public void setEdgeType(int edgeType) {
        this.edgeType = edgeType;
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

    public String getInfo() {
        return info;
    }

    public void setInfo(String info) {
        this.info = info;
    }

    public boolean isEqual() {
        return isEqual;
    }

    public void setEqual(boolean equal) {
        isEqual = equal;
    }

    public int getFrom() {
        return from;
    }

    public void setFrom(int from) {
        this.from = from;
    }

    public int getTo() {
        return to;
    }

    public void setTo(int to) {
        this.to = to;
    }

    @Override
    public String toString() {
        String edgeString = "Edge{edgeType=" + edgeType2String(edgeType) + ", from=" + from + ", to=" + to;
        if (edgeType == 2) {
            edgeString += ", start=" + start +
                    ", end=" + end;
        }
        if (edgeType == 3) {
            if (info != null && !info.equals("")) {
                edgeString += ", info='" + info + '\'';
            }
        }
        if (edgeType == 4) {
            if (isEqual) {
                edgeString += ",equal=" + isEqual;
            }
        }


        edgeString += '}';
        return edgeString;
    }

    public JSONObject toJSON() {

        JSONObject edgeJSON = new JSONObject();
        edgeJSON.put("from", from);
        edgeJSON.put("to", to);
        edgeJSON.put("start", start);
        edgeJSON.put("end", end);
        edgeJSON.put("edgeType", edgeType);
        edgeJSON.put("isEqual", isEqual);
        edgeJSON.put("info", info);

        return edgeJSON;

    }
}
