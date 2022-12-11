package cn.edu.nju.ws.edgqa.utils.linking;

import cn.edu.nju.ws.edgqa.domain.beans.Link;
import cn.edu.nju.ws.edgqa.utils.FileUtil;
import org.apache.jena.base.Sys;
import org.json.JSONArray;
import org.json.JSONObject;

import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;

public class EntityLinkingThread implements Callable<Map<String, List<Link>>> {
    public static final int LINKING_DEXTER = 0;
    public static final int LINKING_EARL = 1;
    public static final int LINKING_FALCON = 2;
    private final String nodeStr;
    private int linkingTool = -1;

    public EntityLinkingThread(int linkingTool, String nodeStr, Map<String, List<Link>> resultMap) {
        this.linkingTool = linkingTool;
        this.nodeStr = nodeStr;
    }

    @Override
    public Map<String, List<Link>> call() throws Exception {
        Map<String, List<Link>> res = null;
        if (linkingTool == LINKING_DEXTER) {
            res = LinkingTool.getDexterLinking(nodeStr);
        } else if (linkingTool == LINKING_EARL) {
            res = LinkingTool.getEARLLinking(nodeStr);
        } else if (linkingTool == LINKING_FALCON) {
            res = LinkingTool.getFalconLinking(nodeStr);
        }
        return res;
    }


    public static void main(String[] args) {
        JSONArray data = new JSONArray(FileUtil.readFileAsString("src/main/resources/datasets/lcquad-test.json"));
        JSONObject dexterRes = new JSONObject();
        JSONObject earlRes = new JSONObject();
        for (int idx = 0; idx < data.length(); idx++) {
            String question = data.getJSONObject(idx).getString("corrected_question");

            Map<String, List<Link>> dexterLink = LinkingTool.getDexterLinking(question);
            try {
                Thread.sleep(300);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            Map<String, List<Link>> earlLink = LinkingTool.getEARLLinking(question);

            System.out.println("[question " + idx + "] " + question);
            dexterRes.put(Integer.toString(idx), dexterLink);
            earlRes.put(Integer.toString(idx), earlLink);
            System.out.println();
            if (idx % 10 == 0) {
                FileUtil.writeStringToFile(dexterRes.toString(), "query_logs/dexter_link.json");
                FileUtil.writeStringToFile(earlRes.toString(), "query_logs/earl_link.json");
            }
        }
    }
}
