package cn.edu.nju.ws.edgqa.utils;

import cn.edu.nju.ws.edgqa.domain.beans.relation_detection.Paraphrase;
import cn.edu.nju.ws.edgqa.handler.Detector;

import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class ParaphraseUtil {

    /**
     * Get predicate list, given a piece of text
     *
     * @param mention a piece of text
     * @return a set of predicate, it maybe null
     */
    public static Set<String> getPredicateList(String mention) {
        if (mention == null || mention.isEmpty()) return null;
        Set<String> predicateList = new HashSet<>();
        Map<String, List<Paraphrase>> dict = Detector.getParaphraseMap();

        for (Map.Entry<String, List<Paraphrase>> entry : dict.entrySet()) {
            for (Paraphrase paraphrase : entry.getValue()) {
                if (paraphrase.getParaphraseWords().contains(mention)) {  // use contains for now
                    predicateList.add(paraphrase.getPredicate());
                }
            }
        }
        return predicateList;
    }

    /**
     * Get a list of paraphrase words, given a predicate
     *
     * @param predicate a predicate string
     * @return a set of paraphrase words, it maybe null
     */
    public static Set<String> getParaphraseOfPredicate(String predicate) {
        Set<String> res = new HashSet<>();

        List<Paraphrase> paraphraseList = Detector.getParaphraseMap().get(predicate);
        if (paraphraseList == null || paraphraseList.isEmpty()) return null;
        for (Paraphrase paraphrase : paraphraseList) {
            res.add(paraphrase.getParaphraseWords());
        }
        return res;
    }

    public static Set<String> getParaphraseOfMention(String mention) {
        Set<String> res = new HashSet<>();

        Set<String> predicateList = getPredicateList(mention);
        if (predicateList == null)
            return null;
        for (String predicate : predicateList) {
            Set<String> paraphraseSet = getParaphraseOfPredicate(predicate);
            if (paraphraseSet != null) {
                res.addAll(paraphraseSet);
            }
            res.add(UriUtil.extractUri(predicate));
        }
        return res;
    }
}
