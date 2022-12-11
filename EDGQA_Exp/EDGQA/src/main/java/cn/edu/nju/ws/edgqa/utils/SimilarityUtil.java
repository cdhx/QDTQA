package cn.edu.nju.ws.edgqa.utils;


import cn.edu.nju.ws.edgqa.domain.beans.Link;
import cn.edu.nju.ws.edgqa.main.QAArgs;
import cn.edu.nju.ws.edgqa.handler.QuestionSolver;
import cn.edu.nju.ws.edgqa.utils.semanticmatching.NeuralSemanticMatchingScorer;
import cn.edu.nju.ws.edgqa.utils.similarity.*;
import org.jetbrains.annotations.NotNull;

import java.util.*;


public class SimilarityUtil {

    /**
     * Key: mention||candidate; value: similarity score
     */
    private static Map<String, Double> relationSimilarityCache = new HashMap<>();

    public static Map<String, Double> getRelationSimilarityCache() {
        return relationSimilarityCache;
    }

    public static void setRelationSimilarityCache(Map<String, Double> relationSimilarityCache) {
        SimilarityUtil.relationSimilarityCache = relationSimilarityCache;
    }

    /**
     * get the similarity of a mention and a set of candidates
     *
     * @param mention    mention
     * @param candidates set of candidates
     * @return map of score. key: candidate; value: score
     */
    public static HashMap<String, Double> getBertSetSimilarity(String mention, Set<String> candidates) {
        // score map
        HashMap<String, Double> res = new HashMap<>();

        if (QAArgs.isUsingRelationSimilarityCache()) { // similarity cache
            Iterator<String> iter = candidates.iterator();
            while (iter.hasNext()) {
                String candidate = iter.next();
                String key = mention + "||" + candidate;
                if (relationSimilarityCache.containsKey(key)) {
                    res.put(candidate, relationSimilarityCache.get(key));
                    iter.remove();
                }
            }
        }

        if (!candidates.isEmpty()) {
            Map<String, Double> relationScoreMap = NeuralSemanticMatchingScorer.relation_semantic_matching_score(mention, candidates);
            if (QAArgs.isUsingRelationSimilarityCache()) { // update cache
                for (String candidate : relationScoreMap.keySet()) {
                    relationSimilarityCache.put(mention + "||" + candidate, relationScoreMap.get(candidate));
                }
            }
            res.putAll(relationScoreMap);
        }
        return res;
    }

    /**
     * get the lexical and dictionary Similarity of a set of candidates
     *
     * @param mention    mention
     * @param candidates a set of candidates
     * @return map of score. key: candidate; value: score
     */
    public static HashMap<String, Double> getLexDictSetSimilarity(String mention, Set<String> candidates) {

        HashMap<String, Double> simRes = new HashMap<>();
        for (String candidate : candidates) {
            if (!simRes.containsKey(candidate)) {
                simRes.put(candidate, getScoreWithParaphrase(candidate, mention));
            }
        }
        return simRes;
    }

    public static HashMap<String, Double> getDictSetSimilarity(String mention, Set<String> candidates) {
        HashMap<String, Double> simRes = new HashMap<>();
        for (String candidate : candidates) {
            if (!simRes.containsKey(candidate)) {
                simRes.put(candidate, getScoreWithParaphrase(candidate, mention));
            }
        }
        return simRes;
    }

    /**
     * get the lexical Similarity of a set of candidates
     *
     * @param mention    mention
     * @param candidates a set of candidates
     * @return map of score. key: candidate; value: score
     */
    public static HashMap<String, Double> getLexicalSetSimilarity(String mention, Set<String> candidates) {

        HashMap<String, Double> simRes = new HashMap<>();
        for (String candidate : candidates) {
            if (!simRes.containsKey(candidate)) {
                simRes.put(candidate, getScore(candidate, mention));
            }
        }
        return simRes;
    }

    /**
     * get the composite Similarity(Bert and lexical) of a set of candidates
     *
     * @param mention    mention
     * @param candidates a set of candidates
     * @return map of score. key: candidate; value: score
     */
    public static HashMap<String, Double> getCompositeSetSimilarity(String mention, Set<String> candidates) {
        long startTime = System.currentTimeMillis();
        HashMap<String, Double> bertSetSimilarity = getBertSetSimilarity(mention, candidates);
        HashMap<String, Double> lexDictSetSimilarity = getLexDictSetSimilarity(mention, candidates);

        HashMap<String, Double> res = new HashMap<>();

        for (String key : lexDictSetSimilarity.keySet()) {
            double lexScore = lexDictSetSimilarity.get(key);
            if (!bertSetSimilarity.containsKey(key)) {
                res.put(key, (lexScore));
            }
            double bertScore = bertSetSimilarity.get(key);
            if (!res.containsKey(key)) {
                //res.put(key,Math.max(lexScore,bertScore));
                res.put(key, (lexScore * 0.55 + bertScore * 0.45));
            }
        }

        long timeCount = System.currentTimeMillis() - startTime;
        synchronized (QuestionSolver.lock) {
            Timer.setTotalSimilarityTime(Timer.getTotalSimilarityTime() + timeCount);
        }
        return res;
    }

    /**
     * get the lexical similarity (average of three score) between str1 and str2, case sensitive
     *
     * @param str1 str1
     * @param str2 str2
     * @return case sensitive lexical similarity
     */
    public static double getScore(@NotNull String str1, @NotNull String str2) {
        SimilarityStrategy strategy = new JaroWinklerStrategy();
        SimilarityStrategy strategy1 = new LevenshteinDistanceStrategy();
        SimilarityStrategy strategy2 = new DiceCoefficientStrategy();

        StringSimilarityService service = new StringSimilarityServiceImpl(strategy);
        str1 = str1.trim();
        str2 = str2.trim();
        double score = service.score(str1, str2);
        double score1 = new StringSimilarityServiceImpl(strategy1).score(str1, str2);
        double score2 = new StringSimilarityServiceImpl(strategy2).score(str1, str2);

        return (score + score1 + score2) / 3.0;
    }

    /**
     * get the lexical similarity (average of three score) between str1 and str2, case insensitive
     *
     * @param str1 str1
     * @param str2 str2
     * @return case insensitive lexical similarity
     */
    public static double getScoreIgnoreCase(@NotNull String str1, @NotNull String str2) {
        return getScore(str1.toLowerCase(), str2.toLowerCase());
    }


    /**
     * get the semantic similarity by dictionary between str1 and str2
     *
     * @param label   relation label
     * @param mention surface form
     * @return similarity by dictionary
     */
    public static double getDictScore(String label, String mention) {


        return 0.0;
    }


    /**
     * calculate the similarity score between label and mention based on RL Dict
     *
     * @param label   relation Label
     * @param mention mention in the sentence
     * @return similarity score
     */
    public static double getScoreWithParaphrase(String label, String mention) {
        long startTime = System.currentTimeMillis();
        if (label == null || mention == null)
            return 0;
        label = label.replaceAll(" +", " ").trim();
        mention = mention.replaceAll(" +", " ").trim();

        // lexical similarity between label and mention
        double score = getScore(label, mention);
        String predicateLabel = label;
        if (label.contains(" ")) { // contain space, convert to CamelCase
            predicateLabel = UriUtil.toCamelCase(label);
        }
        score = Double.max(score, getScore(predicateLabel, mention));

        //lexical similarity
        double lexicalScore = score;

        Set<String> paraphraseSet = ParaphraseUtil.getParaphraseOfPredicate(predicateLabel);
        // not in the dictionary, return lexical similarity score
        if (paraphraseSet == null || paraphraseSet.isEmpty()) return score;
        for (String paraphrase : paraphraseSet) {//Calculate the similarity between mention and paraphrase
            score = Double.max(score, getScore(paraphrase, mention));
        }

        //semantic similarity
        double paraphraseScore = score;

        //Reverse score will cause semantic drift
        /*paraphraseSet = ParaphraseUtil.getParaphraseOfMention(mention);
        if (paraphraseSet == null || paraphraseSet.isEmpty()) return score;
        for (String paraphrase : paraphraseSet) {
            score = Double.max(score, getScore(label, paraphrase));
        }*/
        long timeCount = System.currentTimeMillis() - startTime;
        synchronized (QuestionSolver.lock) {
            Timer.setTotalSimilarityTime(Timer.getTotalSimilarityTime() + timeCount);
        }

        //return Double.max(lexicalScore, semanticScore);
        return (lexicalScore + paraphraseScore) / 2;
    }

    /**
     * judge if the nodeStr is a duplicate refer
     *
     * @param nodeStr nodeStr like `is #entity1`
     * @param resKey  resKey like `#entity1`
     * @return whether it is a duplicate refer
     */
    public static boolean isDescriptionEqual(String nodeStr, String resKey) {

        String str1 = nodeStr.replaceAll("(am|is|are|was|were) ", "")
                .replaceAll("(has|have|had) ", "")
                .replaceAll("(?i)(among|within)", "")
                .replaceAll("(?i)(one|members) of", "")
                .replaceAll("(?i)all", "")
                .replaceAll("((is|are|was|were) )?the ((total|whole) )?(name|number) of", "")
                .replaceAll(" +", " ").trim();
        System.out.println("Str1:" + str1);

        return str1.equals(resKey);
    }

    /**
     * Give the confidence that the input string is an entity mention
     *
     * @param mention a piece of string
     * @param map     linking map
     * @return the confidence
     */
    public static double getMentionConfidence(@NotNull String mention, @NotNull Map<String, List<Link>> map) {
        double score = 0.0;
        if (!map.containsKey(mention)) {
            return score;
        }
        List<Link> linkList = map.get(mention);
        if (linkList != null && linkList.size() > 0) {
            linkList.sort(Collections.reverseOrder());
            score = linkList.get(0).getScore();
        }

        if (score > 0) {
            // judge if a string could be an entity mention
            if (mention.matches("\\d*")) { // a series of number
                score *= 0.01;
            }
            if (mention.matches("[a-zA-z]{1}")) { // a single character (probably not a meaningful word)
                score *= 0.01;
            }
            if (mention.matches("[a-zA-z+\\-_]{0,2}")) { // short string and symbols
                score *= 0.01;
            }
        }
        return score;
    }

    /**
     * Get the coverage score of one hop properties of an entity against all the other entities
     * for the topic mention selection
     *
     * @param oneHopPropertyLabels the one-hop properties of one entity
     * @param entityLinkingKeySet  the whole key set of entity linking
     * @return the coverage score
     */
    public static double getRelationCoverage(Set<String> oneHopPropertyLabels, Set<String> entityLinkingKeySet) {
        double score = 0.0;
        int cnt = 1;
        for (String property : oneHopPropertyLabels) {
            for (String mention : entityLinkingKeySet) {
                if (property.toLowerCase().contains(mention) || mention.toLowerCase().contains(property)) {
                    score += 1;
                }
            }
        }
        return score / cnt;
    }

    public static void main(String[] args) {
        System.out.println(getCompositeSetSimilarity("be cofound by", new HashSet<>(Arrays.asList("founder"))));
    }
}
