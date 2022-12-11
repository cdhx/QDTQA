package cn.edu.nju.ws.edgqa.handler;

import cn.edu.nju.ws.edgqa.main.QAArgs;
import cn.edu.nju.ws.edgqa.utils.CacheUtil;
import cn.edu.nju.ws.edgqa.utils.SimilarityUtil;

import java.io.IOException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class QASystem {
    /**
     * Thread pool for entity linking
     */
    protected static ExecutorService pool = Executors.newFixedThreadPool(12);

    protected static void postProcess() throws IOException {
        pool.shutdown(); // shutdown the thread pool
        if (QAArgs.isCreatingLinkingCache()) {
            CacheUtil.getDexterOutput().writeObject(CacheUtil.getDexterMap());
            CacheUtil.getFalconOutput().writeObject(CacheUtil.getFalconMap());
            CacheUtil.getEarlOutput().writeObject(CacheUtil.getEarlMap());
            System.out.println("[INFO] The entity linking cache have been saved");
        }
        if (QAArgs.isCreatingRelationSimilarityCache()) {
            CacheUtil.getRelationSimilarityOutput().writeObject(SimilarityUtil.getRelationSimilarityCache());
            System.out.println("[INFO] The relation similarity cache have been saved");
        }
    }
}
