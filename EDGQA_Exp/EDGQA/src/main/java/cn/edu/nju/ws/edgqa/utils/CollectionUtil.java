package cn.edu.nju.ws.edgqa.utils;

import cn.edu.nju.ws.edgqa.domain.beans.Link;
import cn.edu.nju.ws.edgqa.utils.enumerates.LinkEnum;

import java.util.*;
import java.util.stream.Collectors;

public class CollectionUtil {

    // take intersection of two String set
    public static Set<String> setIntersect(Set<String> set1, Set<String> set2) {
        Set<String> newSet = new HashSet<>(set1.size());
        newSet.addAll(set1);
        newSet.retainAll(set2);
        return newSet;
    }

    // merge two HashMap，merge srcMap to destMap, type indicate:true=entity false=relation
    public static void mergeLinkMap(Map<String, List<Link>> srcMap, Map<String, List<Link>> destMap, LinkEnum type) {

        for (String key : srcMap.keySet()) {
            List<Link> linkList = srcMap.get(key);
            //key = key.toLowerCase(); remain cased or uncased in original
            if (linkList != null && linkList.size() > 0 && linkList.get(0).getType() == type) {//same type
                boolean hasKey = false;
                for (String oldKey : destMap.keySet()) {
                    if (oldKey.toLowerCase().equals(key.toLowerCase())) {
                        linkList.forEach(link -> link.setMention(oldKey));
                        destMap.get(oldKey).addAll(linkList);
                        hasKey = true;
                        break;
                    }
                }
                if (!hasKey) {
                    destMap.put(key, new ArrayList<>(linkList));
                }
            }
        }

    }

    public static void sortLinkMap(HashMap<String, List<Link>> map) {
        for (String key : map.keySet()) {
            List<Link> linkList = map.get(key);
            linkList.sort(Collections.reverseOrder());
        }
    }

    public static List<Object> removeDuplicateElements(List<Object> objectList) {
        return objectList.stream().distinct().collect(Collectors.toList());
    }
}
