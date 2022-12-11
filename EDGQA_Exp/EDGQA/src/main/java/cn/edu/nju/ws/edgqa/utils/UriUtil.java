package cn.edu.nju.ws.edgqa.utils;

import cn.edu.nju.ws.edgqa.utils.kbutil.KBUtil;
import com.google.common.base.CaseFormat;
import org.jetbrains.annotations.NotNull;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class UriUtil {
    /**
     * Directly split the label from uri, split by space, there may be Uppercase or CamelCase
     *
     * @param uri the URI of the knowledge base, camelCase or underline connected
     * @return the label of this URI, maybe uppercase or CamelCase, underline is replaced by space
     */
    public static String splitLabelFromUri(String uri) {
        String[] split = uri.split("/");
        String label = split[split.length - 1];
        label = label.replaceAll("_", " ");
        return label;
    }

    public static String[] splitLabelFromUri_Freebase(String uri) {
        String[] split = uri.split("/");
        String domain_type_property = split[split.length - 1];
        domain_type_property = domain_type_property.replace("_", " ").trim();

        return domain_type_property.split("\\.");

    }

    /**
     * Extract the label from a knowledge base URI
     *
     * @param uri the URI of the knowledge base, camelCase or underline connected
     * @return the label of this URI, lowercase, split by space
     */
    public static String extractUri(String uri) {
        String[] splitList = uri.split("/");
        if (splitList.length == 0) return "";
        String label = splitList[splitList.length - 1].trim();
        label = label.replaceAll("_", " ");
        label = CaseFormat.LOWER_CAMEL.to(CaseFormat.LOWER_UNDERSCORE, label);
        label = label.replaceAll("_", " ").toLowerCase();
        label = label.replaceAll(" +", " ");
        return label;
    }

    /**
     * convert a label in NL to the url label in KB
     *
     * @param label label in NL e.g. known for
     * @return label in KB as CamelCase e.g. knownFor
     */
    public static String toCamelCase(@NotNull String label) {
        if (!label.contains(" ") && !label.contains("_"))
            return label;
        label = label.trim();
        if (label.isEmpty()) return label;
        StringBuilder stringBuilder = new StringBuilder(label);
        stringBuilder.setCharAt(0, Character.toLowerCase(label.charAt(0)));
        label = stringBuilder.toString();
        label = label.replaceAll(" ", "_");
        label = CaseFormat.LOWER_UNDERSCORE.to(CaseFormat.LOWER_CAMEL, label);
        return label;
    }

    /**
     * extract the labels of a list of uri, return a Map<Label,List<URI>>
     *
     * @param uriList a list of uri
     * @return a Map<Label,List<URI>>
     */
    public static Map<String, List<String>> extractLabelMap(List<String> uriList) {
        Map<String, List<String>> labelMap = new HashMap<>();
        for (String uri : uriList) {
            if (uri.length() > 64) continue; // the uri is too long

            String label = KBUtil.queryLabel(uri);
            if (label == null) {
                label = extractUri(uri);
            }
            if (!labelMap.containsKey(label)) {
                labelMap.put(label, new ArrayList<>());
            }

            labelMap.get(label).add(uri);
        }
        return labelMap;
    }

    /**
     * Find all the dbo and dbp in the question set file.
     *
     * @param questionSetFilePath the file path of question set
     */
    public static void predicateFiltering(String questionSetFilePath) throws IOException {
        Set<String> predicateSet = new HashSet<>();

        File questionSetFile = new File(questionSetFilePath);
        InputStreamReader isr = new InputStreamReader(new FileInputStream(questionSetFile), StandardCharsets.UTF_8);
        BufferedReader bufferedReader = new BufferedReader(isr);
        String line = null;
        Pattern p1 = Pattern.compile("<http://dbpedia.org/ontology/.*?>");
        Pattern p2 = Pattern.compile("<http://dbpedia.org/property/.*?>");
        while ((line = bufferedReader.readLine()) != null) {
            Matcher m1 = p1.matcher(line);
            while (m1.find()) {
                predicateSet.add(m1.group().replace("<", "").replace(">", ""));
            }
            Matcher m2 = p2.matcher(line);
            while (m2.find()) {
                predicateSet.add(m2.group().replace("<", "").replace(">", ""));
            }
        }
        for (String predicate : predicateSet) {
            if (predicate.startsWith("http://dbpedia.org/ontology/")) {
                if (Character.isLowerCase(predicate.charAt(28)))
                    System.out.println(predicate);
            } else {
                System.out.println(predicate);
            }
        }
    }

    public static void entityFiltering(String questionSetFilePath) throws IOException {
        Set<String> entitySet = new HashSet<>();
        Map<String, String> charMap = buildCharMap("dataset/dict/char.csv");
        File questionSetFile = new File(questionSetFilePath);
        InputStreamReader isr = new InputStreamReader(new FileInputStream(questionSetFile), StandardCharsets.UTF_8);
        BufferedReader bufferedReader = new BufferedReader(isr);
        String line = null;
        Pattern p1 = Pattern.compile("<http://dbpedia.org/ontology/.*?>");
        Pattern p2 = Pattern.compile("<http://dbpedia.org/resource/.*?>");
        while ((line = bufferedReader.readLine()) != null) {
            Matcher m1 = p1.matcher(line);
            while (m1.find()) {
                String newEntity = replaceWithMap(charMap, m1.group().replace("<", "").replace(">", ""));
                entitySet.add(newEntity);
            }
            Matcher m2 = p2.matcher(line);
            while (m2.find()) {
                String newEntity = replaceWithMap(charMap, m2.group().replace("<", "").replace(">", ""));
                entitySet.add(newEntity);
            }
        }
        for (String entity : entitySet) {
            System.out.println(entity);
        }
        bufferedReader.close();
    }

    public static Map<String, String> buildCharMap(String charFilePath) throws IOException {
        Map<String, String> charMap = new HashMap<>();
        BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(charFilePath), StandardCharsets.UTF_8));
        String line = null;
        while ((line = br.readLine()) != null) {
            String[] lineSplit = line.split(",");
            if (lineSplit.length != 2) continue;
            charMap.put(lineSplit[0], lineSplit[1]);
        }
        br.close();
        return charMap;
    }

    public static String replaceWithMap(Map<String, String> map, String str) {
        for (Map.Entry<String, String> entry : map.entrySet()) {
            str = str.replace(entry.getKey(), entry.getValue());
        }
        return str;
    }

    public static void main(String[] args) throws IOException {
        entityFiltering("src/main/resources/datasets/lcquad-all.json");
    }
}
