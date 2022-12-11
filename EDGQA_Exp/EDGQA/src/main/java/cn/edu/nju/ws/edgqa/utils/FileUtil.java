package cn.edu.nju.ws.edgqa.utils;

import cn.edu.nju.ws.edgqa.domain.beans.relation_detection.Paraphrase;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class FileUtil {

    /**
     * read text file, return as String
     *
     * @param filePath path of the file
     * @return file as a String
     */
    public static String readFileAsString(String filePath) {
        StringBuilder sb = new StringBuilder();
        BufferedReader bfr = null;
        try {
            bfr = new BufferedReader(new FileReader(filePath));
            String line = null;
            while ((line = bfr.readLine()) != null) {
                sb.append(line).append("\n");
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                if (bfr != null) {
                    bfr.close();
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return sb.toString();

    }

    /**
     * read text file, return as String
     *
     * @param filePath the file path
     * @return a list of string
     */
    public static List<String> readFileAsList(String filePath) {
        File input = new File(filePath);
        List<String> result = new ArrayList<>();
        BufferedReader bfr = null;
        try {
            bfr = new BufferedReader(new FileReader(input));
            String line;
            while ((line = bfr.readLine()) != null) {
                if (!line.isEmpty()) {
                    result.add(line);
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                if (bfr != null) {
                    bfr.close();
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        return result;
    }

    /**
     * write a string to file
     *
     * @param strToWrite the string to write
     * @param filePath   the file path to write in
     */
    public static void writeStringToFile(String strToWrite, String filePath) {
        try {
            BufferedWriter bfw = new BufferedWriter(new FileWriter(filePath));
            bfw.write(strToWrite);
            bfw.flush();
            bfw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * write a String List to file
     *
     * @param strListToWrite the string list to write
     * @param filePath       the file path to write in
     */
    public static void writeStringListtoFile(List<String> strListToWrite, String filePath) {
        try {
            BufferedWriter bfw = new BufferedWriter(new FileWriter(filePath));
            for (String str : strListToWrite) {
                bfw.write(str + "\n");
            }
            bfw.flush();
            bfw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    /**
     * Load paraphrase from a file
     *
     * @param filePath      the path of dict file
     * @param splitRegex    the regex for splitting
     * @param readFirstLine true if the first line is the paraphrase data
     * @return the list of paraphrases
     */
    public static List<Paraphrase> loadParaphrase(String filePath, String splitRegex, boolean readFirstLine) throws IOException {
        List<Paraphrase> paraphraseList = new ArrayList<>();
        BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(filePath)));
        String line;
        if (!readFirstLine) {
            line = br.readLine();
        }
        while ((line = br.readLine()) != null) {
            String[] lineSplit = line.split(splitRegex);
            if (lineSplit.length == 2) {
                Paraphrase paraphrase = new Paraphrase(lineSplit[0], lineSplit[1]);
                paraphraseList.add(paraphrase);
            } else if (lineSplit.length == 3) {
                Paraphrase paraphrase = new Paraphrase(lineSplit[0], lineSplit[1], Double.parseDouble(lineSplit[2]));
                paraphraseList.add(paraphrase);
            } else {
                br.close();
                return paraphraseList;
            }
        }
        br.close();
        return paraphraseList;
    }
}
