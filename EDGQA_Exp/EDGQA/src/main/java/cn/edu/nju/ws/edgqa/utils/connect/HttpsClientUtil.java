package cn.edu.nju.ws.edgqa.utils.connect;

import org.apache.http.HttpEntity;
import org.apache.http.HttpHost;
import org.apache.http.NameValuePair;
import org.apache.http.client.config.RequestConfig;
import org.apache.http.client.entity.UrlEncodedFormEntity;
import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.conn.ssl.SSLConnectionSocketFactory;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClientBuilder;
import org.apache.http.message.BasicNameValuePair;
import org.apache.http.ssl.SSLContextBuilder;
import org.apache.http.util.EntityUtils;
import org.json.JSONObject;

import javax.net.ssl.SSLContext;
import java.util.ArrayList;
import java.util.List;

public class HttpsClientUtil {

    private static CloseableHttpClient httpClient;

    static {
        try {

            SSLContext sslContext = SSLContextBuilder.create().useProtocol(SSLConnectionSocketFactory.SSL).loadTrustMaterial((x, y) -> true).build();
            //HttpHost proxy = new HttpHost("127.0.0.1", 7890, "http");  // 7890 is default clash proxy port
            //RequestConfig config = RequestConfig.custom().setConnectTimeout(5000).setSocketTimeout(5000).setConnectionRequestTimeout(5000).setProxy(proxy).build();
            RequestConfig config = RequestConfig.custom().setConnectTimeout(30000).setSocketTimeout(30000).setConnectionRequestTimeout(30000).build();
            httpClient = HttpClientBuilder.create().setDefaultRequestConfig(config).setSSLContext(sslContext).setSSLHostnameVerifier((x, y) -> true).build();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static String doPost(String url, String jsonString) {
        try {
            HttpPost httpPost = new HttpPost(url);
            StringEntity stringEntity = new StringEntity(jsonString, "utf-8");
            stringEntity.setContentType("application/json");
            httpPost.setEntity(stringEntity);
            httpPost.setHeader("Content-Type", "application/json");
            CloseableHttpResponse response = httpClient.execute(httpPost);
            int statusCode = response.getStatusLine().getStatusCode();
            if (statusCode != 200) {
                httpPost.abort();
                throw new RuntimeException("HttpClient, error status code :" + statusCode);
            }
            HttpEntity entity = response.getEntity();
            String result = null;
            if (entity != null) {
                result = EntityUtils.toString(entity, "utf-8");
            }
            EntityUtils.consume(entity);
            response.close();
            return result;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    public static String doPostWithParams(String url, JSONObject obj) {
        try {
            HttpPost httpPost = new HttpPost(url);

            // parameters
            List<NameValuePair> nvps = new ArrayList<>();
            for (String key : obj.keySet()) {
                nvps.add(new BasicNameValuePair(key, obj.getString(key)));
            }
            httpPost.setEntity(new UrlEncodedFormEntity(nvps));

            httpPost.setHeader("Content-Type", "application/json");
            CloseableHttpResponse response = httpClient.execute(httpPost);
            int statusCode = response.getStatusLine().getStatusCode();
            if (statusCode != 200) {
                httpPost.abort();
                throw new RuntimeException("HttpClient, error status code :" + statusCode);
            }
            HttpEntity entity = response.getEntity();
            String result = null;
            if (entity != null) {
                result = EntityUtils.toString(entity, "utf-8");
            }
            EntityUtils.consume(entity);
            response.close();
            return result;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    public static String doGet(String url) {
        try {
            HttpGet httpGet = new HttpGet(url);
            //StringEntity stringEntity = new StringEntity(jsonString, "utf-8");
            //stringEntity.setContentType("application/json");
            httpGet.setHeader("accept", "application/json");
            //httpGet.setEntity(stringEntity);
            CloseableHttpResponse response = httpClient.execute(httpGet);
            int statusCode = response.getStatusLine().getStatusCode();
            if (statusCode != 200) {
                httpGet.abort();
                response.close();
                throw new RuntimeException("HttpClient,error status code :"
                        + statusCode);
            }
            HttpEntity entity = response.getEntity();
            String result = null;
            if (entity != null) {
                result = EntityUtils.toString(entity, "utf-8");
            }
            EntityUtils.consume(entity);
            response.close();
            return result;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }
}
