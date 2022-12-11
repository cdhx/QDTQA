package cn.edu.nju.ws.edgqa.utils.semanticmatching;

import cn.edu.nju.ws.edgqa.domain.edg.EDG;
import cn.edu.nju.ws.edgqa.main.QAArgs;
import cn.edu.nju.ws.edgqa.utils.connect.HttpsClientUtil;
import cn.edu.nju.ws.edgqa.utils.enumerates.DatasetEnum;
import cn.edu.nju.ws.edgqa.utils.enumerates.KBEnum;
import cn.edu.nju.ws.edgqa.utils.kbutil.KBUtil;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import org.jetbrains.annotations.NotNull;
import org.json.JSONArray;

import java.util.*;


public class NeuralSemanticMatchingScorer {

    private static final String serverIP = "114.212.190.19";

    // port for relation semantic matching and block query reranking
    private static int relation_semantic_matching_serverPort = 5682;
    private static final int block_query_reranking_lcquad_serverPort = 5683;
    private static final int block_query_reranking_qald_serverPort = 5684;

    // service url for relation semantic matching and block query reranking
    private static String relation_semantic_matching_url = "http://" + serverIP + ":" + relation_semantic_matching_serverPort + "/relation_detection";
    private static String block_query_reranking_url = "http://" + serverIP + ":" + block_query_reranking_lcquad_serverPort + "/query_rerank";

    public static Map<String, Double> relation_semantic_matching_score(@NotNull String question, @NotNull Set<String> labels) {

        if (EDG.getKB() == KBEnum.Freebase) {
            relation_semantic_matching_serverPort = 5680;
            relation_semantic_matching_url = "http://" + serverIP + ":" + relation_semantic_matching_serverPort + "/relation_detection";
        }

        String[] labelArr = new String[labels.size()];
        labels.toArray(labelArr);
        Double[] detection_res = null;
        Map<String, Double> resMap = new HashMap<>();
        try {
            JSONArray array = new JSONArray(labelArr);

            String input = "{\"question\": \"" + question + "\", \"labels\": " + array + "}";
            String output = HttpsClientUtil.doPost(relation_semantic_matching_url, input);
            Gson gson = new Gson();
            Map<String, Double[]> map = gson.fromJson(String.valueOf(output), new TypeToken<Map<String, Double[]>>() {
            }.getType());
            detection_res = map.get("detection_res");

        } catch (Exception e) {
            e.printStackTrace();
        }

        if (detection_res != null) {
            for (int i = 0; i < detection_res.length; i++) {
                resMap.put(labelArr[i], detection_res[i]);
            }
        }

        return resMap;
    }

    public static Map<String, Double> query_reranking_score(@NotNull String question, @NotNull Set<String> labels) {

        String[] labelArr = new String[labels.size()];
        labels.toArray(labelArr);
        Double[] detection_res = null;
        Map<String, Double> resMap = new HashMap<>();
        if (QAArgs.getDataset() == DatasetEnum.QALD_9) {
            block_query_reranking_url = "http://" + serverIP + ":" + block_query_reranking_qald_serverPort + "/query_rerank";
        }
        try {
            JSONArray array = new JSONArray(labelArr);

            String input = "{\"edg_block\": \"" + question + "\", \"sparql_queries\": " + array + "}";

            String output = HttpsClientUtil.doPost(block_query_reranking_url, input);
            Gson gson = new Gson();
            Map<String, Double[]> map = gson.fromJson(String.valueOf(output), new TypeToken<Map<String, Double[]>>() {
            }.getType());
            detection_res = map.get("rerank_res");

        } catch (Exception e) {
            e.printStackTrace();
        }

        if (detection_res != null) {
            for (int i = 0; i < detection_res.length; i++) {
                resMap.put(labelArr[i], detection_res[i]);
            }
        }

        return resMap;
    }
    
    public static void main(String[] args) {

        KBUtil.init(DatasetEnum.LC_QUAD);
        Map<String, Double> score = query_reranking_score("[BLK]  [DES] Name #entity1 [DES] is Ptolemy XIII Theos Philopator [DES]  [BLK]  [DES] a queen [DES] whose parent is Ptolemy XII Auletes [DES] whose parent is consort", new HashSet<>(Arrays.asList(" [TRP] ?e1 parent Ptolemy XII Auletes [TRP] ?e0 name ?e1")));
        Map<String, Double> score_1 = query_reranking_score("[BLK]  [DES] are the bands associated with #entity1 [BLK]  [DES] the artists of My Favorite Girl", new HashSet<>(Arrays.asList("\t [TRP] ?e1 Artist My Favorite Girl (Dave Hollister song) [TRP] ?e0 associated musical artist ?e1")));
        System.out.println(score);
        System.out.println(score_1);
        
        Map<String, Double> score_2 = relation_semantic_matching_score("moon", new HashSet<>(Arrays.asList("satellite","moon","sun")));
        System.out.println(score_2);

    }
}
