package cn.edu.nju.ws.edgqa.utils;

import cn.edu.nju.ws.edgqa.domain.beans.TreeNode;
import cn.edu.nju.ws.edgqa.utils.linking.LinkingTool;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.util.CoreMap;

import java.io.IOException;
import java.util.*;

public class NLPUtil {

    private static final HashSet<String> entityTypes = new HashSet<>(Arrays.asList("PERSON", "LOCATION", "ORGANIZATION", "MISC"));


    private static final StanfordCoreNLP parsePipeline;
    private static final StanfordCoreNLP lemmatizePipeline;
    private static final StanfordCoreNLP nerPipeline;
    private static final StanfordCoreNLP tokenizePipeline;
    private static final StanfordCoreNLP posPipeline;

    static {

        // properties for constituency parsing
        Properties parseProps = new Properties();
        parseProps.setProperty("annotators", "tokenize, ssplit, pos, parse");
        //props.setProperty("parse.model", "edu/stanford/nlp/models/srparser/englishSR.ser.gz");
        parsePipeline = new StanfordCoreNLP(parseProps);

        // properties for lemmatization
        Properties lemmaProps = new Properties();
        lemmaProps.setProperty("annotators", "tokenize,ssplit,pos,lemma");
        lemmatizePipeline = new StanfordCoreNLP(lemmaProps);

        // properties for NER
        Properties nerProps = new Properties();
        nerProps.setProperty("annotators", "tokenize,ssplit,pos,lemma,ner");
        nerProps.setProperty("ner.applyFineGrained", "false");
        nerPipeline = new StanfordCoreNLP(nerProps);

        // properties for tokenization
        Properties tokenizeProps = new Properties();
        tokenizeProps.setProperty("annotators", "tokenize");
        tokenizePipeline = new StanfordCoreNLP(tokenizeProps);

        // properties for POS tagging
        Properties props = new Properties();
        props.setProperty("annotators", "tokenize,ssplit,pos");
        posPipeline = new StanfordCoreNLP(props);

    }



    /**
     * return the lemmatization of a word
     *
     * @param word raw word
     * @return lemmatization of a word
     */
    public static String getLemma(String word) {

        if (word.equalsIgnoreCase("born")) {
            return word;
        }

        Annotation doc = new Annotation(word);
        lemmatizePipeline.annotate(doc);
        return doc.get(CoreAnnotations.TokensAnnotation.class).get(0).lemma();
    }

    /**
     * return the lemmatization of a sentence
     *
     * @param sent raw sentence
     * @return lemmatization of the sentence
     */
    public static String getLemmaSent(String sent) {
        ArrayList<String> tokens = getTokens(sent);
        StringBuilder sb = new StringBuilder();
        for (String token : tokens) {
            sb.append(getLemma(token)).append(" ");
        }
        return sb.toString().trim();
    }

    /**
     * return all the entity mentions spotted by coreNLP ner module in a sentence
     * @param sentence natural language sentence
     * @return all the entity mentions spotted
     */
    public static List<String> coreNLPEntityMentions(String sentence) {
        List<String> res = new LinkedList<>();

        CoreDocument document = new CoreDocument(sentence);
        nerPipeline.annotate(document);
        for (CoreEntityMention mention : document.entityMentions()) {
            if (entityTypes.contains(mention.entityType())) {
                res.add(mention.text());
            }
        }

        return res;
    }

    /**
     * judge whether a phase is an Entity by dexter and coreNLP
     * @param phase a natural language phase
     * @return if it is an entity, return true, else return false
     */
    public static boolean judgeIfEntity(String phase) {
        System.out.println("[DEBUG] judgeIfEntity:" + phase);

        if (phase == null || phase.equals("")) {  // prevent it's an empty string
            return false;
        }

        if (phase.matches("<e\\d>")) {
            return true;
        }

        double e = 0.7; // if len(mention) / len(phase) is greater than 0.7, it is identified as an entity

        // A dexter2 server is needed, comment it if the server is not available
        if (LinkingTool.isDexterEntity(phase, e)) {
            return true;
        }

        for (String mention : coreNLPEntityMentions(phase)) {
            if ((double) mention.length() / phase.length() >= e) {
                return true;
            }
        }

        return false;
    }

    /**
     * get a list of the tokens in a sentence by coreNLP tokenizer
     * @param sentence natural language sentence
     * @return a list of the tokens in the sentence
     */
    public static ArrayList<String> getTokens(String sentence) {

        //null string check
        if (sentence == null || sentence.trim().equals("")) {
            return new ArrayList<>();
        }

        Annotation annotation = new Annotation(sentence);
        tokenizePipeline.annotate(annotation);

        ArrayList<String> result = new ArrayList<>();
        List<CoreLabel> coreLabels = annotation.get(CoreAnnotations.TokensAnnotation.class);

        for (CoreLabel token : coreLabels) {
            result.add(token.toString());
        }

        return result;
    }

    /**
     * get the constituency parsing of a sentence by coreNLP
     * @param sentence natural language sentence
     * @return constituency parsing result
     */
    public static String getSyntaxTree(String sentence) {

        Annotation document = new Annotation(sentence);
        parsePipeline.annotate(document);
        List<CoreMap> sentences = document.get(CoreAnnotations.SentencesAnnotation.class);
        StringBuilder sb = new StringBuilder();
        for (CoreMap sent : sentences) {
            sb.append(sent.get(TreeCoreAnnotations.TreeAnnotation.class).toString());
        }
        return sb.toString();
    }

    /**
     * get the POS tags of the tokens in a sentence
     * @param sentence natural language sentence
     * @return the POS tags of the tokens
     */
    public static List<String> getPOS(String sentence) {

        Annotation annotation = new Annotation(sentence);
        posPipeline.annotate(annotation);
        LinkedList<String> poses = new LinkedList<>();
        for (CoreLabel token : annotation.get(CoreAnnotations.TokensAnnotation.class)) {
            //poses.add(token.get(CoreAnnotations.PartOfSpeechAnnotation.class))
            poses.add(token.tag());
        }
        return poses;
    }

    //Judge whether a phrase is a verb phrase, return true if yes, false if not

    /**
     * judge whether a span of the sentence is a verb-phrase
     * @param span a span of the sentence in the form of String
     * @return if the sentence is a verb-phrase, return true; else return false
     */
    public static boolean judgeIfVP(String span) {
        HashSet<String> relationTags = new HashSet<>(
                Arrays.asList("VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "VP", "S", "SQ", "ADJP", "ADVP"));
        String syntaxTree = NLPUtil.getSyntaxTree(span);
        TreeNode treeNode = TreeNode.createTree(syntaxTree);

        if (treeNode.getFirstChild().getData().trim().equals("FRAG")) {//If it is FRAG, go down one level
            treeNode = treeNode.getFirstChild();
        }

        if (!treeNode.children.isEmpty()) {
            TreeNode treeNode1 = treeNode.getFirstChild();
            return relationTags.contains(treeNode1.data.trim());
        }
        return false;

    }

    /**
     * remove the redundant head and tail in a sentence for question preprocess
     * @param str sentence
     * @return a sentence with redundant head and tail trimmed
     */
    public static String removeRedundantHeaderAndTailer(String str) {
        String originalStr = str;
        int len = str.length();
        try {
            if (str.endsWith(" of")) {
                if (str.startsWith("be ")) {
                    str = str.substring(3, len - 3);
                } else if (str.startsWith("the ")) {
                    str = str.substring(4, len - 3);
                }
            } else if (str.endsWith(" in")) {
                if (str.startsWith("whose ") || str.startsWith("which ")) {
                    str = str.substring(6, len - 3);
                }
            }
        } catch (IndexOutOfBoundsException e) {
            e.printStackTrace();
            return originalStr;
        }
        return str;
    }

    public static String transferParentheses(String s) {
        return s.replace("(", "[").replace(")", "]");
    }

    public static void main(String[] args) throws IOException {

        //String question = "How many different teams have the players debuted in Houston Astros played for?";
        //String syntaxTree = NLPUtil.getSyntaxTree(question);
        //System.out.println(transferParentheses(syntaxTree));

    }


}

