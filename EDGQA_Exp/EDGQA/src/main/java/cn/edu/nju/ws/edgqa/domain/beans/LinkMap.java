package cn.edu.nju.ws.edgqa.domain.beans;

import cn.edu.nju.ws.edgqa.utils.kbutil.KBUtil;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class LinkMap {
    private Map<String, List<Link>> data;

    /**
     * Default constructor
     */
    public LinkMap() {
        this.data = new HashMap<>();
    }

    /**
     * Constructor
     *
     * @param data the linking map data
     */
    public LinkMap(Map<String, List<Link>> data) {
        this.data = data;
    }

    public Map<String, List<Link>> getData() {
        return data;
    }

    public void setData(Map<String, List<Link>> data) {
        this.data = data;
    }

    public Link topLink() {
        double highestScore = -1;
        Link res = null;
        for (String key : data.keySet()) {
            for (Link link : data.get(key)) {
                if (link.getScore() > highestScore) {
                    highestScore = link.getScore();
                    res = link;
                }
            }
        }
        return res;
    }

    public Link oneHopTopLink(String entityURI) {
        Set<String> oneHopRelationSet = KBUtil.oneHopProperty(entityURI);
        double hightestScore = -1;
        Link res = null;
        for (String key : data.keySet()) {
            for (Link link : data.get(key)) {
                if (oneHopRelationSet.contains(link.getUri())) {
                    if (link.getScore() > hightestScore) {
                        hightestScore = link.getScore();
                        res = link;
                    }
                }
            }
        }
        return res;
    }
}
