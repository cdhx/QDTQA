package cn.edu.nju.ws.edgqa.domain.edg;

import org.jetbrains.annotations.NotNull;

import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static cn.edu.nju.ws.edgqa.domain.staticres.WordList.getImperativeTrigger;
import static cn.edu.nju.ws.edgqa.domain.staticres.WordList.judgeIfImperative;


public class Preprocessor {

    public static String preProcessQuestion(String question) {

        if (judgeIfImperative(question)) {
            if (!Objects.equals(getImperativeTrigger(question), "list")) { //list maybe identified incorrectly
                question = question.replaceAll("\\?$", ".");
            }
        } else { // not imperative
            if (!question.matches(".*[?.!]$")) {
                question = question + "?";
            }
        }
        question = question.trim();

        // substitution of some common forms of modal verbs
        question = question.replaceAll("can be said to be ", "is ");
        question = question.replaceAll("can be said as ", "is ");
        question = question.replaceAll("can be", "is ");


        // substitution of some common meaningless phrases
        question = question.replaceAll("a list of ", " ");
        question = question.replaceAll("(?i)(what (kind|sort|genre) of )", "what ");
        question = question.replaceAll("(?i)(which (kind|sort|genre) of )", "what ");
        question = question.replaceAll("(?i)(count the (.*)number of)", "count the");


        // substitution of some common typos
        question = question.replaceAll("aldo ", "also ");
        question = question.replaceAll("(?i)which ", "which ");
        question = question.replaceAll("(?i)whihc ", "which ");
        question = question.replaceAll("(?i)which\\w ", "which ");
        question = question.replaceAll("(?i)whos ", "who is");
        question = question.replaceAll("/'", "?");
        question = question.replaceAll("Whichlocation", "Which location");
        question = question.replaceAll("palce", "place");
        question = question.replaceAll(" it's ", " its ");

        // substitution of some acronyms to help with entity identification
        question = question.replaceAll("(?i)( mt )", " mountain ");

        // substitution of questions lead by prepositions, e.g., In which war did Roh Tae Woo and Lee Leffingwell fight
        String regex = "(?i)(^(in|on|to|from|under|for|through|by) (which|where|when|what|who|whose|how|whom) (.*))";
        Pattern pattern1 = Pattern.compile(regex);
        Matcher matcher1 = pattern1.matcher(question.trim());
        if (matcher1.matches()) {
            question = matcher1.group(3) + " " + matcher1.group(4);
        }

        // capitalize the first letter
        question = question.trim();
        question = question.substring(0, 1).toUpperCase() + question.substring(1);

        return question;
    }

    public static String rewriteQALDSparql(@NotNull String goldenQueryStr) {
        String res = goldenQueryStr;
        if (goldenQueryStr.contains("ASK WHERE"))
            return res;
        if (goldenQueryStr.contains("SELECT DISTINCT")) {
            res = res.replaceAll("SELECT DISTINCT", "SELECT DISTINCT (");
        } else if (goldenQueryStr.contains("SELECT COUNT(DISTINCT")) {
            res = res.replace("SELECT COUNT(DISTINCT", "SELECT COUNT (DISTINCT(");
        } else if (goldenQueryStr.contains("SELECT ")) {
            res = res.replaceAll("SELECT ", "SELECT (");
        }
        res = res.replaceAll(" WHERE", ") WHERE");
        return res;
    }
}
