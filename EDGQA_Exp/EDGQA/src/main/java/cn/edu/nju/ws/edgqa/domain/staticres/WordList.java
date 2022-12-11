package cn.edu.nju.ws.edgqa.domain.staticres;


import cn.edu.nju.ws.edgqa.domain.beans.TreeNode;
import cn.edu.nju.ws.edgqa.utils.NLPUtil;
import cn.edu.nju.ws.edgqa.utils.enumerates.QueryType;

import java.util.*;

import static cn.edu.nju.ws.edgqa.domain.beans.TreeNode.createTree;

public class WordList {
    public static List<String> be_form = Arrays.asList("be", "is", "was", "are", "were", "being", "'s");
    public static List<String> have_form = Arrays.asList("have", "had", "has");
    public static List<String> neg_form = Arrays.asList("not", "n't");
    public static List<String> do_form = Arrays.asList("did", "do", "does", "done");
    public static LinkedHashSet<String> imperativeWordSet = new LinkedHashSet<>(
            Arrays.asList("count", "give me a count of", "give me the total number of", "total number of", "give me", "give", "list", "show me", "show", "name", "find", "tell me", "tell"));
    public static LinkedHashSet<String> whWordSet = new LinkedHashSet<>(
            Arrays.asList("what", "which", "where", "whose", "whom", "who", "when", "how many", "how much", "how")
    );
    public static LinkedHashSet<String> whTagSet = new LinkedHashSet<>(
            Arrays.asList("WDT", "WP", "WP$", "WRB")
    );
    public static LinkedHashSet<String> countWordSet = new LinkedHashSet<>(
            Arrays.asList("give me a count of", "give me the total number of", "total number of", "count", "how many", "what is the number of", "how much")
    );
    public static LinkedHashSet<String> generalWordSet = new LinkedHashSet<>(
            Arrays.asList("do", "did", "does", "am", "is", "are", "was", "were",
                    "can you", "will you", "shall you", "could you", "would you", "should you",
                    "can", "will", "shall", "could", "would", "should")
    );
    /**
     * be word list, ordered
     */
    public static LinkedHashSet<String> beWordSet = new LinkedHashSet<>(
            Arrays.asList("being", "be", "am", "is", "are", "was", "were")
    );
    //auxiliaryWordSet, ordered
    public static LinkedHashSet<String> auxiliaryWordSet = new LinkedHashSet<>(
            Arrays.asList("does", "did", "do")
    );
    // tags that indicate a relation, ordered
    public static HashSet<String> relationTags = new LinkedHashSet<>(
            Arrays.asList("VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "VP", "ADJP", "ADVP", "CONJP", "PP"));

    public static boolean isContainList(List<String> list, String s) {
        boolean ok = false;
        for (String t : list) {
            if (s.equals(t)) {
                ok = true;
                break;
            }
        }
        return ok;
    }

    public static QueryType getCoarseQuestionType(String question) {
        if (judgeIfImperative(question)) {
            return QueryType.LIST;
        }
        if (judgeIfSpecial(question)) {
            if (question.toLowerCase().contains("how many") || question.toLowerCase().contains("how much")) {
                return QueryType.COUNT;
            }
            return QueryType.COMMON;
        }
        if (judgeIfSpecial(question)) {
            return QueryType.JUDGE;
        }
        return QueryType.COMMON;
    }

    public static boolean judgeIfAuxiliary(String word) {
        return auxiliaryWordSet.contains(word.trim().toLowerCase());
    }

    public static boolean judgeIfContainAux(String sentence) {
        for (String word : auxiliaryWordSet) {
            if (sentence.toLowerCase().contains(word + " ")) {
                return true;
            }
        }
        return false;
    }

    public static boolean judgeIfImperative(String question) {
        for (String word : imperativeWordSet) {
            if (question.trim().toLowerCase().startsWith(word)) {
                return true;
            }
        }
        return false;
    }

    public static String getImperativeTrigger(String question) {
        for (String word : imperativeWordSet) {
            if (question.trim().toLowerCase().startsWith(word)) {
                return word;
            }
        }
        return null;
    }

    public static boolean judgeIfCount(String question) {
        for (String word : countWordSet) {
            if (question.trim().toLowerCase().startsWith(word)) {
                return true;
            }
        }
        return false;
    }

    public static String getCountTrigger(String question) {
        for (String word : countWordSet) {
            if (question.trim().toLowerCase().startsWith(word)) {
                return word;
            }
        }
        return null;
    }

    public static boolean judgeIfGeneral(String question) {
        for (String word : generalWordSet) {
            if (question.trim().toLowerCase().startsWith(word)) {
                return true;
            }
        }
        return false;
    }

    public static boolean judgeIfSpecial(String question) {
        for (String word : whWordSet) {
            if (question.trim().toLowerCase().startsWith(word)) {
                return true;
            }
        }
        return false;
    }

    public static String getGeneralTrigger(String question) {
        for (String word : generalWordSet) {
            if (question.trim().toLowerCase().startsWith(word)) {
                return word;
            }
        }
        return null;
    }

    public static String getWHTrigger(String question) {
        for (String word : whWordSet) {
            if (question.trim().toLowerCase().startsWith(word)) {
                return word;
            }
        }
        return null;
    }

    public static boolean judgeIfBeWord(String word) {
        return beWordSet.contains(word.trim().toLowerCase());
    }

    public static boolean judgeIfContainBeWord(String sentence) {
        for (String beWord : beWordSet) {
            if (sentence.toLowerCase().contains(beWord + " ")) {
                return true;
            }
        }
        return false;
    }

    public static String getWHWord(String sentence) {
        for (String whWord : whWordSet) {
            if (sentence.trim().toLowerCase().startsWith(whWord)) {
                return whWord;
            }
        }
        return null;
    }

    /**
     * judge whether a word is a wh-word
     *
     * @param word word
     * @return true=wh-word; false o.w.
     */
    public static boolean judgeIfWHWord(String word) {
        return whWordSet.contains(word.toLowerCase().trim());
    }

    /**
     * given a word Tag, judge if it is a wh-word
     *
     * @param wordTag pos tag of the word
     * @return true=the word is a wh-word; false o.w.
     */
    public static boolean judgeIFWHTag(String wordTag) {
        return whTagSet.contains(wordTag.trim());
    }

    /**
     * judge whether span may contain a relation
     *
     * @param span string text
     * @return true for yes and false for no
     */
    public static boolean containsRelation(String span) {

        if (span == null) {
            return false;
        }

        //empty string
        if (span.trim().equals("")) {
            return false;
        }

        if (judgeIfBeWord(span.trim())) {
            return false;
        }

        if (judgeIfAuxiliary(span.trim())) {
            return false;
        }

        if (judgeIfWHWord(span.trim())) {
            return false;
        }

        if (span.matches("(.*) of the (.*)")) {
            return true;
        }

        if (span.matches("(.*) as (.*)")) {
            return true;
        }

        if (span.matches("(.*) by (.*)")) {
            return true;
        }


        String syntaxTree = NLPUtil.getSyntaxTree(span);
        TreeNode treeNode = createTree(syntaxTree);

        LinkedList<TreeNode> toSearch = new LinkedList<>();
        toSearch.push(treeNode);
        while (!toSearch.isEmpty()) {
            TreeNode pop = toSearch.pop();
            if (relationTags.contains(pop.data.trim())) {
                return true;
            }

            if (!pop.children.isEmpty()) {
                toSearch.addAll(pop.children);
            }
        }

        return false;
    }
}
